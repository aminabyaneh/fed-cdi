"""
    File name: utils.py
    Author: Amin Abyaneh
    Email: aminabyaneh@gmail.com
    Date created: 25/04/2021
    Python Version: 3.8
    Description: Methods and classes required for the project.
"""

# ========================================================================
# Copyright 2021, The CFL Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================

import glob
import os.path
import random
import itertools

from typing import List, Dict, Tuple
from networkx.algorithms.shortest_paths.generic import shortest_path

import numpy as np
import pandas as pd
import networkx as nx

from logging_settings import logger

DEFAULT_OBSERVATION_SIZE = 5000


def split_variables_set(num_vars, accessible_percentages, seed=0):
    """Split a set of variables based on accessible percentages.

    Args:
        num_vars (int): Total number of variables.
        accessible_percentages (List[int]): Accessible percentage for each client.
        seed (int, optional): Random seed for shuffling. Defaults to 0.

    Returns:
        List: A list of splits.
    """
    variables = [var for var in range(num_vars)]

    random.seed(seed)
    random.shuffle(variables)

    indices = list()
    start, end = 0, 0
    for per in accessible_percentages:
        start = end
        end = start + int((per / 100) * num_vars)
        indices.append(variables[start: end])

    return indices


def find_shortest_distance_dict(variable_index, adjacency_mat) -> Dict:
    """ Get a dict with elements that indicate the shortest distance between
    each variable and the one determined by index.

    Args:
        variable_index (int): Reference variable for distance measuring.
        adjacency_mat (np.ndarray): Graph adjacency matrix.

    Returns:
        Dict: The distance dictionary.
    """

    graph = nx.DiGraph(adjacency_mat)
    distance_dict: Dict[int, int] = dict()

    shortest_path_result = nx.shortest_path(graph, variable_index)

    for var_idx in range(adjacency_mat.shape[0]):
        if var_idx in shortest_path_result.keys():
            distance_dict[var_idx] = len(shortest_path_result[var_idx]) - 1
        else:
            distance_dict[var_idx] = np.inf

    return distance_dict


def calculate_metrics(predicted_mat: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """Returns a dictionary with detailed metrics comparing the current prediction to
    the ground truth graph.

    Note: calculations are the same as ENCO module but with numpy.

    Args:
        predicted_mat (np.ndarray): The predicted adjacency matrix.
        ground_truth (np.ndarray): The true structure of underlying causality graph.

    Returns:
        Dict: Metrics dictionary.
    """
    false_positives = np.logical_and(predicted_mat, np.logical_not(ground_truth))
    false_negatives = np.logical_and(np.logical_not(predicted_mat), ground_truth)

    TP = np.logical_and(predicted_mat, ground_truth).astype(float).sum()
    TN = np.logical_and(np.logical_not(predicted_mat), np.logical_not(ground_truth)).astype(float).sum()
    FP = false_positives.astype(float).sum()
    FN = false_negatives.astype(float).sum()
    TN = TN - predicted_mat.shape[-1]

    recall = TP / max(TP + FN, 1e-5)
    precision = TP / max(TP + FP, 1e-5)

    rev = np.logical_and(predicted_mat, ground_truth.T)
    num_revs = rev.astype(float).sum()
    SHD = (false_positives + false_negatives + rev + rev.T).astype(float).sum() - num_revs

    metrics = {
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "SHD": int(SHD),
        "reverse": int(num_revs),
        "recall": recall,
        "precision": precision,
    }

    return metrics


def build_experimental_dataset(adjacency_matrix: np.ndarray, causal_order: List,
                               assignment: Dict[int, List[str]] or Dict[int, float],
                               assignment_type: str = 'observation_assignment',
                               file_name: str = 'experimental_data',
                               noise_distribution: str = 'uniform',
                               observation_size: int = DEFAULT_OBSERVATION_SIZE,
                               verbose: bool = False):
    """
    Build an experimental dataset to be used in inference techniques and
    local clients of distributed network.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix of the data.
        assignment (Dict[int: List[str]] or Dict[int, float]): Provides the data assignment dictionary.
        assignment_type (str): This option can be either 'variable_assignment' or 'observation_assignment'.
        causal_order (List[int]): The causal order of variables in the adjacency matrix.
        file_name (str): The name of the data when saved as a file.
        noise_distribution (str): The noise distribution, defaults to 'uniform'.
        observation_size (int): The total number of observations to be generated.
        verbose (bool): Show details about dataset after build or not. Defaults to False.
    """

    data = np.zeros((len(causal_order), observation_size))

    n_dist = None
    if noise_distribution == 'uniform':
        n_dist = np.random.uniform
    elif noise_distribution == 'gaussian':
        n_dist = np.random.gaussian

    for variable_index in causal_order:
        if n_dist is not None:
            data[variable_index] += n_dist(size=observation_size)

        for element in causal_order[:causal_order.index(variable_index)]:
            data[variable_index] += adjacency_matrix[variable_index][element] * data[element]

    data_frame = pd.DataFrame(data=data.transpose(),
                              index=range(observation_size),
                              columns=['X' + str(i) for i in range(len(causal_order))])

    data_frame.to_csv(path_or_buf=os.path.join(os.pardir, 'data', file_name + '_dataset.csv'), index=False)
    save_data_object(adjacency_matrix, file_name + '_adjacency_matrix', 'data')
    save_data_object(assignment, file_name + f'_{assignment_type}', 'data')

    if verbose:
        logger.info(f'Dataset generated: \n Name: {file_name} \t Size: {data_frame.shape} '
                    f'\t Additive Noise: {noise_distribution}\n')

    return data_frame


def generate_accessible_percentages(number_of_clients: int,
                                    minimum_accessible_percentage: float,
                                    maximum_accessible_percentage: float,
                                    is_randomized: bool = False) -> Dict[int, float]:
    """
    This generates randomized accessible percentages for a fixed number of clients. The results can only
    be used in 'observation_assignment' mode of the datasets.

    Args:
        number_of_clients (int): The total number of clients who want to have access.
        minimum_accessible_percentage (float): Maximum possible access percentage.
        maximum_accessible_percentage (float): Minimum possible access percentage.
        is_randomized (bool): If set true, the percentages would be randomized, otherwise
        a sweep between minimum and maximum is provided.

    Returns:
        Dict[int, float]: The percentages associated with clients.
    """

    assignment_dictionary: Dict[int, float] = dict()

    if not is_randomized:
        access_values_list = \
            np.linspace(minimum_accessible_percentage,
                        maximum_accessible_percentage, num=number_of_clients)

        assignment_dictionary = \
            {client_id: accessible_percentage
             for client_id, accessible_percentage in enumerate(access_values_list)}

    else:
        for client_id in range(number_of_clients):
            assignment_dictionary[client_id] = random.uniform(minimum_accessible_percentage,
                                                              maximum_accessible_percentage)

    return assignment_dictionary


def weighted_matrix_to_binary(matrix: np.ndarray) -> np.ndarray:
    """
    Convert an adjacency matrix which is organized for a weighted graph into a
    binary one, usually to use scoring techniques.

    Args:
        matrix (np.ndarray): The input weighted adjacency matrix.

    Returns:
        matrix (np.ndarray): Binary adjacency matrix.
    """

    binary_matrix: np.ndarray = matrix > 0.5
    return binary_matrix.astype(int)


def save_data_object(input_object, file_name: str, save_directory: str):
    """
    Save the passed dictionary.

    Args:
        input_object: Dictionary to be stored.
        file_name (str): Name of the file to be stored.
        save_directory (str): Directory of the stored file relative to the main project folder.
        Defaults to Output folder.
    """

    np.save(os.path.join(save_directory, file_name + '.npy'), input_object)


def load_data_object(file_name: str, save_directory: str):
    """
    Save the passed dictionary.

    Args:
        file_name (str): Name of the file to be loaded.
        save_directory (str): Directory of the loaded file relative to the main project folder.

    Returns:
        Object: The loaded object from a given directory and file.
    """

    return np.load(os.path.join(save_directory, file_name + '.npy'), allow_pickle=True)


def retrieve_dsdi_stored_data(dir: str, experiment_id: int, round_id: int):
    """
    Load all the information stored by DSDI clients in the work directory of
    the DSDI sub-module.

    Args:
        experiment_id (int): The id of experiment folder.
        round_id (int): Identifier determining the round number for which the data must be retrieved.

    Returns:
        np.ndarray: An array containing adjacency matrix and accessible percentage.
    """

    load_dir = f'{dir}/experiment_{experiment_id}/Gamma_Data_{round_id}_*'

    for filename in glob.glob(load_dir):
        yield np.load(filename, allow_pickle=True)


def resume_dsdi_experiments(basename: str = 'experiment_') -> int:

    # Check whether a re-start job has occurred in the cluster
    load_dir = 'work/'

    existing_experiments_ids = list()
    for filename in glob.glob(f'{load_dir}{basename}*'):
        id_str = filename.replace(f'{load_dir}{basename}', '')
        existing_experiments_ids.append(int(id_str))

    # Redo the last and perhaps unfinished experiment
    start_from = 0

    if len(existing_experiments_ids) != 0:
        logger.info(f'Found the following experiments: {existing_experiments_ids}')
        start_from = max(existing_experiments_ids)

    return start_from


def generate_npy_prior_matrix(matrix: np.ndarray = None,
                              dimensions: Tuple = (3, 3), file_name: str = 'prior_info',
                              directory: str = 'CausalLearningFederated/data/'):
    """
    Generating a sample prior matrix to test basic functionalities of the integrated repository.

    Args:
        matrix (np.ndarray): The matrix object in case a non-random matrix is needed. If blank,
        a random matrix is generated.

        dimensions (Tuple): The dimensions of the adjacency matrix.
        file_name (str): Name of the file to be saved.
        directory (str): The directory of the saved file.

    Returns:
        np.ndarray: The randomized matrix of priors.
    """

    if matrix is None:
        mat = np.random.uniform(low=0, high=1, size=dimensions)
    else:
        mat = matrix

    logger.info(f'Generated prior information is: {mat}')
    save_data_object(mat, file_name=file_name, save_directory=directory)

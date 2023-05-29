"""
    File name: integration_scripts.py
    Author: Amin Abyaneh
    Email: aminabyaneh@gmail.com
    Date created: 15/06/2021
    Python Version: 3.8
    Description: Tests and integrations related to incorporating the Rosemary algorithm in
    the federated setup.
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

import os
from typing import Tuple

import numpy as np
import pandas as pd

from logging_settings import logger
from utils import save_data_object


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


def store_sample_dataset(save_directory: str, seed: int = 0, model: str = 'chain3', num_cats: int = 2,
                         observation_samples_count: int = 1000, intervention_samples_count: int = 1000):
    """
    Execute the command to store the entire dataset information in to a chosen directory.

    Note: Only works if being run on the DSDI project directory, nested in the 'libs' folder.

    Args:
        save_directory (str): Address of the saved data.
        seed (int): Random seed.
        model (str): Choose a model for the dataset. Must comply with the model you choose when running the
        algorithm itself.

        num_cats (int): Specifies the number of categories for the categorical distribution.
        observation_samples_count (int): The number of observational samples.
        intervention_samples_count (int): The number of samples per single-node intervention.
    """

    command = f'python run.py dump mixedData --seed {seed} ' \
              f'--dump-dir={save_directory} ' \
              f'-p {model} ' \
              f'--num-cats {num_cats} ' \
              f'--num-samples-obs {observation_samples_count} ' \
              f'--num-samples-int {intervention_samples_count}'

    os.system(command=command)


def load_sample_dataset(data_directory: str):
    """
    Load the entire dataset dumped by the last command.

    Args:
        data_directory (str): Address of the dataset directory.

    Returns:
        np.ndarray, pd.DataFrame, pd.DataFrame:
            adjacency matrix, interventional, and observational datasets.
    """

    # Return the adjacency matrix
    adjacency_matrix = np.genfromtxt(os.path.join(data_directory, 'dag.csv'), delimiter=',')
    yield adjacency_matrix

    # Return the interventional dataframe
    interventional_df = pd.read_csv(os.path.join(data_directory, 'dataInt.csv'), index_col=False,
                                    header=None, names=[f'X{i}' for i in range(adjacency_matrix.shape[0])])

    int_data = [int(i) for i in np.genfromtxt(os.path.join(data_directory, 'dataInt_expId.csv'))]
    interventional_expId = pd.Series(int_data, name='I')
    yield pd.concat([interventional_df, interventional_expId], axis=1)

    # Return the observational dataframe
    observational_df = pd.read_csv(os.path.join(data_directory, 'dataObs.csv'), index_col=0, header=None).T
    observational_df.columns = [f'X{i}' for i in range(adjacency_matrix.shape[0])]
    yield observational_df


if __name__ == '__main__':
    # run the functions here
    # generate_npy_prior_matrix(matrix=np.asarray([[0, 0, 0], [0.8, 0, 0], [0, 0.9, 0]]))

    # dataset related operations
    os.chdir('../libs/CausalLearningFederated')
    store_sample_dataset(save_directory='data/dataset')
    mat, df_int, df_obs = load_sample_dataset(data_directory='data/dataset')
    print(df_int.head(15))
    print(df_obs.head(15))

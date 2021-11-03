"""
    File name: causal_learning.py
    Author: Amin Abyaneh
    Email: aminabyaneh@gmail.com
    Date created: 15/04/2021
    Python Version: 3.8
    Description: Implementation of causal learning algorithms.
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

import os, sys

import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from typing import List
from abc import ABC, abstractmethod
from networkx import DiGraph, circular_layout

from logging_settings import logger

sys.path.append("../")

from causal_discovery.enco import ENCO
from causal_graphs.graph_definition import CausalDAG
from causal_graphs.graph_definition import CausalDAGDataset
from causal_graphs.graph_generation import generate_categorical_graph, get_graph_func
from causal_graphs.variable_distributions import _random_categ

class InferenceAlgorithm(ABC):
    """
    Abstract class for a variety of causal learning methods. Each implementation of new
    algorithms should inherit from the InferenceAlgorithm object.

    """

    def __init__(self, verbose: bool = False):
        """
        Initialize a InferenceAlgorithm object.
        """
        np.random.seed(100)

    @abstractmethod
    def build_local_dataset(self):
        """
        Load only a part of observations/variables from a global dataset to implement locality
        and privacy of data.
        """

    @abstractmethod
    def infer_causal_structure(self):
        """
        Develop a causal inference technique using this function.
        """


class DSDIAlg(InferenceAlgorithm):
    """
    Wrapper implementation for the DSDI algorithm introduced by Rosemary et. al.

    The proposed method has the ability to incorporate a prior belief matrix at the beginning of
    each iteration, which makes it a perfect candidate for a federated setup where the model is
    getting updated and- hopefully- improved over the iterations.

    Note: The class is built upon the causal_learning_unknown_interventions github repository:
            https://github.com/nke001/causal_learning_unknown_interventions

    Note: A forked version of this repository may be found in the libs folder.
    """

    def __init__(self):
        """
        Initialize a DSDI Algorithm class.

        Note: Before running this function, you have to make sure that conda environment related to
        DSDI algorithm is up and running (named causal_iclr). For this purpose, follow the guidelines
        in the README file of the DSDI repository.

        """

        super().__init__()

    def load_local_dataset(self, accessible_data: List[str] or float,
                           assignment_type: str = 'observation_assignment',
                           dataset_name: str = "sachs", import_from_directory: bool = False):
        """
        Note: This is currently disabled in this class since the local dataset is distributed online,
        rather than how it is handled in other classes by distribution of a DataFrame and CSV files.
        """

        pass

    def infer_causal_structure(self, accessible_percentage: int = 100, num_clients: int = 5,
                               client_id: int = 0, round_id: int = 0,
                               experiment_id: int = 0, seed: int = 0, num_epochs: int = 50,
                               dpe: int = 10, train_functional: int = 6000, epi_size: int = 10,
                               ipd: int = 100, v: int = 500, gamma_belief: str or None = None,
                               graph: str = 'chain3', store_folder: str = 'default_experiments',
                               verbose: int = 0, predict: int = 0):
        """
        This function uses a system call to run the DSDI run.py file, located in the lib folder as
        a submodule.

        The parameters may be set here, or leave them to the default values instead. For more information, refer
        to DSDI github page addressed in the class description.
        """

        logger.info(f'Entering directory {os.getcwd()}')

        execution_command = f'python run.py train ' \
                            f'--seed {seed} ' \
                            f'--num-epochs {num_epochs} ' \
                            f'--dpe {dpe} ' \
                            f'--train_functional {train_functional} ' \
                            f'--accessible-percentage {accessible_percentage} ' \
                            f'--num-clients {num_clients} ' \
                            f'--client-id {client_id} ' \
                            f'--round-id {round_id} ' \
                            f'--experiment-id {experiment_id} ' \
                            f'--store-folder {store_folder} ' \
                            f'--ipd {ipd} ' \
                            f'--xfer-epi-size {epi_size} ' \
                            f'--mopt adam:5e-2,0.9 ' \
                            f'--gopt adam:5e-3,0.1 ' \
                            f'-v {verbose} ' \
                            f'--lsparse 0.1 ' \
                            f'--bs 256 ' \
                            f'--ldag 0.5 ' \
                            f'--predict {predict} ' \
                            f'--temperature 1 ' \
                            f'--limit-samples 500 ' \
                            f'-N 2 ' \
                            f'-p {graph} '

        if gamma_belief is not None:
            execution_command = execution_command + f'--gammaBelief {gamma_belief}'

        # Execute the training sequence
        logger.info(f'Executing command: \n{execution_command}')
        try:
            os.system(command=execution_command)
        except ModuleNotFoundError:
            logger.critical('Activate conda environment according to DSDI manual!')


class ENCOAlg(InferenceAlgorithm):
    """
    Wrapper implementation for the ENCO algorithm introduced by Philipe et. al.

    The proposed method has the ability to incorporate a prior belief matrix at the beginning of
    each iteration, which makes it a perfect candidate for a federated setup where the model is
    getting updated and- hopefully- improved over the iterations.

    Note: A forked version of this repository is used along with the federated dir.

    Note: Before using this class, you have to make sure that conda environment related to
    ENCO algorithm is up and running (named enco). For this purpose, follow the guidelines
    in the README file of the repository related to creating the environment.
    """

    def __init__(self, client_id: int, external_dataset_dag: CausalDAGDataset,
                 accessible_percentage: int = 100, num_clients: int = 5,
                 int_variables: List[int] or None = None):
        """ Initialize a ENCO Algorithm class.


        Args:
            client_id (int): The unique identifier for the client using this learning module.
            external_dataset_dag (CausalDAGDataset): The global dataset that all the clients have
                access to.

            accessible_percentage (int, optional): The portion of local share accessible to a client.
                Defaults to 100.

            num_clients (int, optional): Total number of clients in the global setup. Defaults to 5.
            int_variables (List[int]orNone, optional): A list of variables for which interventional
                samples are available. Defaults to None.

        Raises:
            ValueError: Check if global dataset is loaded.
        """

        super().__init__()

        # Define orientation and existence mat
        self.inferred_orientation_mat: np.ndarray = np.ndarray([0])
        self.original_adjacency_mat: np.ndarray = np.ndarray([0])
        self.inferred_existence_mat: np.ndarray = np.ndarray([0])
        self.metrics_dict = dict()


        # Initialize federated properties
        self.__client_id = client_id
        self.__accessible_p = accessible_percentage
        self.__int_variables = int_variables

        # Initialize the global dataset
        if external_dataset_dag is None:
            raise ValueError('The global dataset is not loaded!')

        self.original_adjacency_mat = external_dataset_dag.adj_matrix.astype(int)
        self.__data = external_dataset_dag.data_obs
        self.__data_int = external_dataset_dag.data_int

        logger.info(f'Client {self.__client_id} external dataset loaded')
        self.build_local_dataset(num_clients, external_dataset_dag.adj_matrix.shape[0])

    def build_local_dataset(self, num_clients: int, num_vars: int):
        """ Build the local dataset for an specific client.

        Args:
            num_clients (int): The total number of clients in the setup.
            num_vars (int): The number of underlying graph nodes.
        """

        data_length = (self.__data.shape[0] // num_clients)
        start_index = data_length * (self.__client_id)

        data_length_acc = int(data_length * (self.__accessible_p / 100))
        end_index = start_index + data_length_acc - 1

        local_obs_data = self.__data[start_index: end_index]
        logger.info(f'Client {self.__client_id}: Shape of the local observational data: {local_obs_data.shape}')

        local_int_data: np.ndarray = None
        for var_idx in range(num_vars):
            data_length = (self.__data_int.shape[1] // num_clients)
            start_index = data_length * (self.__client_id)

            data_length_acc = int(data_length * (self.__accessible_p / 100))
            end_index = start_index + data_length_acc - 1

            int_sample = self.__data_int[var_idx][start_index: end_index]

            local_int_data = np.array([int_sample]) if local_int_data is None \
                                                    else np.append(local_int_data,
                                                                   np.array([int_sample]),
                                                                   axis=0)
        logger.info(f'Client {self.__client_id}: Shape of the local interventional data: {local_int_data.shape}')

        excluded_variables = [var_idx for var_idx in range(num_vars) if var_idx not in self.__int_variables]
        logger.info(f'Client {self.__client_id}: Excluding following variables: {excluded_variables}\n')

        self._local_dag_dataset = CausalDAGDataset(self.original_adjacency_mat,
                                                   local_obs_data,
                                                   local_int_data,
                                                   exclude_inters=excluded_variables)

    def infer_causal_structure(self, gamma_belief: np.ndarray or None,
                               theta_belief: np.ndarray or None, num_epochs: int = 2):
        """This function calls an inference algorithm using ENCO core functions and class,
        given a dataset_dag.

        The parameters may be set here, or leave them to the default values instead.
        For more information, refer to ENCO github page.

        Args:
            gamma_belief (np.ndarray or None): The prior information on edge existence.
            theta_belief (np.ndarray or None): The prior information on edge orientation.
            num_epochs (int, optional): Total number of epochs for ENCO. Defaults to 2.
        """

        logger.info(f'Client {self.__client_id} started the inference process')
        enco_module = ENCO(graph=self._local_dag_dataset, prior_gamma=gamma_belief,
                           prior_theta=theta_belief)

        if torch.cuda.is_available():
            gpu_name = 'cuda:0'
            enco_module.to(torch.device(gpu_name))

        enco_module.discover_graph(num_epochs=num_epochs)

        self.inferred_orientation_mat = enco_module.get_theta_matrix()
        self.inferred_existence_mat = enco_module.get_gamma_matrix()
        self.binary_adjacency_mat = ((enco_module.get_binary_adjmatrix()).detach().numpy()).astype(int)
        self.metrics_dict = enco_module.get_metrics()

        torch.cuda.empty_cache()
        logger.info(f'Client {self.__client_id} finished the inference process')

    def get_client_id(self):
        """ Getter for client id.

        Returns:
            int: the client id
        """
        return self.__client_id

    def get_accessible_percentage(self):
        """ Getter for the accessible percentage of dataset

        Returns:
            int: the accessible_p variable
        """
        return self.__accessible_p

    def get_interventions_list(self):
        """ Getter for enforced interventions.

        Returns:
            List[int]: variable ids
        """
        return self.__int_variables

    @staticmethod
    def build_global_dataset(obs_data_size: int, int_data_size: int, num_vars: int,
                             graph_type: str, seed: int = 0, num_categs: int = 10,
                             edge_prob: float or None = None) -> CausalDAGDataset:
        """The function builds a graph and an external dataset using soft intervention and
        online sampling from the respective graph.

        Args:
            obs_data_size (int): Size of observational samples.
            int_data_size (int): Size of interventional samples.
            num_vars (int): Number of variables in the graph.
            graph_type (str): Graph type accoring to what is accessible by ENCO now.
            seed (int, optional): Random seed. Defaults to 0.
            num_categs (int, optional): Number of categories for categorical data. Defaults to 10.

            edge_prob (floatorNone, optional): Edge probability in case the graph is defined as "random".
            Defaults to None.

        Returns:
            CausalDAGDataset: A global dataset for other clients to sample from.
        """

        assert graph_type in ["chain", "bidiag", "random", "full", "jungle", "collider"], "Graph not defined."
        graph: CausalDAG = generate_categorical_graph(num_vars=num_vars,
                                                      min_categs=num_categs,
                                                      max_categs=num_categs,
                                                      use_nn=True,
                                                      graph_func=get_graph_func(graph_type),
                                                      edge_prob=edge_prob,
                                                      seed=seed)
        logger.info(f'Graph is built with the provided information: \n {graph}')

        original_adjacency_mat = graph.adj_matrix
        logger.info(f'Global dataset adjacency matrix: \n {original_adjacency_mat.astype(int)}')

        data_obs = graph.sample(batch_size=obs_data_size, as_array=True)
        logger.info(f'Shape of global observational data: {data_obs.shape}')

        data_int = ENCOAlg.sample_int_data(graph, int_data_size)
        logger.info(f'Shape of global interventional data: {data_int.shape}\n')

        return CausalDAGDataset(original_adjacency_mat, data_obs, data_int)

    @staticmethod
    def sample_int_data(graph: CausalDAG, int_data_size: int):
        """ Build an interventional dataset based on the provided parameters.


        Args:
            graph (CausalDAG): The graph for sampling interventins.
            int_data_size (int): Number of samples for interventional data.

        Returns:
            np.ndarray: The interventional dataset.
        """
        data_int: np.ndarray = None

        for var_idx in range(len(graph.variables)):

            # Select variable to intervene on
            var = graph.variables[var_idx]

            # Soft, perfect intervention => replace p(X_n) by random categorical
            # Scale is set to 0.0, which represents a uniform distribution.
            int_dist = _random_categ(size=(var.prob_dist.num_categs,), scale=0.0, axis=-1)

            size = (int_data_size // len(graph.variables)) + 1
            # Sample from interventional distribution
            value = np.random.multinomial(n=1, pvals=int_dist,
                                        size=(size,))
            value = np.argmax(value, axis=-1)

            intervention_dict = {var.name: value}
            int_sample = graph.sample(interventions=intervention_dict,
                                      batch_size=size, as_array=True)

            data_int = np.array([int_sample]) if data_int is None \
                                              else np.append(data_int,
                                                              np.array([int_sample]),
                                                              axis=0)
        return data_int

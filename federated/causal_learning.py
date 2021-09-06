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

import os
from abc import ABC, abstractmethod
from typing import List

import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from cdt.causality.graph import LiNGAM, PC, SAM, GES, GIES
from cdt.data import load_dataset
from networkx import DiGraph, circular_layout

from logging_settings import logger

sys.path.append(os.path.join(os.pardir, 'causal_discovery'))
sys.path.append(os.path.join(os.pardir, 'causal_graphs'))

from causal_discovery.enco import ENCO
from causal_graphs.graph_definition import CausalDAG, CausalDAGDataset
from causal_graphs.graph_generation import generate_categorical_graph, get_graph_func
from causal_graphs.graph_visualization import visualize_graph
from causal_graphs.variable_distributions import _random_categ

class InferenceAlgorithm(ABC):
    """
    Abstract class for a variety of causal learning methods. Each implementation of new
    algorithms should inherit from the CausalModel object.

    Note: The DSDI method  requires some number of iterations to conclude the results,
    also there are some problems related to storing and loading the dataset since they
    only support their own dataset as of now.

    Note: With regard to CDT library, as soon as the setting cdt.SETTINGS.GPU > 0,
    the execution of GPU compatible algorithms will be automatically performed on
    those devices, making the prediction step similar to a traditional algorithm.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize a CausalModel object.
        """

        np.set_printoptions(precision=3, suppress=True)
        np.random.seed(100)

        # Local attributes of CausalModel
        self._verbose = verbose
        self._data: pd.DataFrame = pd.DataFrame()
        self._data_int: pd.DataFrame = pd.DataFrame()
        self._graph: nx.classes.digraph.DiGraph = DiGraph()

        # Global attributes of the CausalModel
        self.original_adjacency_mat: np.ndarray = np.ndarray([0])
        self.inferred_adjacency_mat: np.ndarray = np.ndarray([0])
        self.inferred_DiGraph: nx.classes.digraph.DiGraph = DiGraph()

    def _extract_designated_variables(self, accessible_variables):
        """
        Only keep the accessible part of variables.

        Args:
            accessible_variables (List[str]): Name of the local variables.
        """

        remove_variables_list: List[int] = list()
        for index, col in enumerate(self._data.columns[1:]):
            if col not in accessible_variables:
                remove_variables_list.append(index)

        # Trim the adjacency matrix
        parsed_adjacency_mat = np.delete(arr=nx.to_numpy_matrix(self._graph),
                                         obj=remove_variables_list,
                                         axis=0)

        self.original_adjacency_mat = np.delete(arr=parsed_adjacency_mat,
                                                obj=remove_variables_list,
                                                axis=1)

        # Trim the dataset
        self._data = self._data[accessible_variables]

    def _extract_designated_observations(self, accessible_percentage):
        """
        Only keep the accessible part of observations based on randomness and indicated percentage.

        Args:
            accessible_percentage (float): Name of the local variables.
        """
        # Randomize selection
        self._data = self._data.sample(frac=accessible_percentage / 100)

        # Copy the original adjacency matrix
        self.original_adjacency_mat = nx.to_numpy_array(self._graph)

    def get_observations_size(self) -> int:
        """
        Return the number of observations (rows) in the _data.

        Returns:
            int: The size of observations, or the rows in dataframe.
        """

        return self._data.shape[0]

    def load_local_dataset(self, accessible_data: List[str] or float,
                           assignment_type: str = 'observation_assignment',
                           dataset_name: str = "sachs", import_from_directory: bool = False):
        """
        Load only a part of observations/variables from a global dataset to implement locality
        and privacy of data.

        Args:
            accessible_data (List[str] or float): Choose how much of the dataset can the client see.
            Based on the chosen assignment type, this can be a float (percentage) or a list of accessible,
            variables.

            assignment_type(str): This parameter could be either 'observation_assignment' or
            'variable_assignment'. Defaults tp 'observation_assignment'.

            dataset_name (str): Use a default dataset from CDT library. Defaults to 'sachs'. Alternatively,
            you can import your own dataset as a dataframe by using a url instead of name.

            import_from_directory (bool): If true, the dataset is loaded from a csv file instead of
            a predefined CDT dataset. Use the dataset name for addressing purposes. Defaults to False.
        """
        if import_from_directory:
            # Load a pandas dataframe from a directory
            self._data, self._graph = pd.read_csv(dataset_name), None
        else:
            # Load a default dataset from CDT library
            self._data, self._graph = load_dataset(dataset_name)

        # Determine the assignment type
        if assignment_type == 'observation_assignment':
            self._extract_designated_observations(accessible_percentage=accessible_data)
        else:
            self._extract_designated_variables(accessible_variables=accessible_data)

    def visualize_causal_graph(self, is_original: bool = False,
                               file_name: str = 'dag',
                               directory: str = '../data'):
        """
        Visualize the original or calculated causal graphs.

        Args:
            is_original (bool): Set True if the visualization of original data graph is required.
            Defaults to False.

            file_name (str): Name of the file to be saved. Defaults to dag.
            directory (str): The directory in which the file is saved. Defaults to 'data'.
        """

        plotted_graph = self.inferred_DiGraph
        if not is_original:
            plotted_graph = self._graph

        plt.figure(figsize=[10, 12])
        nx.draw_networkx(plotted_graph, pos=circular_layout(plotted_graph), arrows=True, with_labels=True,
                         arrowstyle='Simple', node_size=2000, font_size=10)

        plt.savefig(os.path.join(directory, file_name + '.png'))
        plt.show()

    @abstractmethod
    def infer_causal_structure(self):
        """
        Develop a causal inference technique using this function.
        """


class LiNGAMAlg(InferenceAlgorithm):
    """
    Wrapper implementation for the LiNGAM algorithm.

    The class is built upon the lingam library for inference and graphviz for visualizations.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize a LiNGAMAlg class.
        """

        super().__init__(verbose=verbose)

    def infer_causal_structure(self):
        """
        Build a direct lingam model and infer the underlying structure.

        The structure is then stored in the class attribute 'inferred_adjacency_mat' for ease
        of access in future.
        """
        # Build and fit a direct lingam model
        lingam_model = LiNGAM()
        self.inferred_DiGraph = lingam_model.predict(self._data)

        # Store the weighted adjacency matrix
        self.inferred_adjacency_mat = nx.to_numpy_array(self.inferred_DiGraph)

        # Display the matrix
        if self._verbose:
            logger.info(f'The inferred weighted adjacency matrix is: \n {self.inferred_adjacency_mat}')


class PCAlg(InferenceAlgorithm):
    """
    Wrapper implementation for the PC algorithm.

    The class is built upon the cdt.causality.graph.PC library.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize a PCAlg class.
        """

        super().__init__(verbose=verbose)

    def infer_causal_structure(self):
        """
        Build a direct lingam model and infer the underlying structure.

        The structure is then stored in the class attribute 'inferred_adjacency_mat' for ease
        of access in future.
        """
        # Build and fit a direct lingam model
        pc_model = PC()
        self.inferred_DiGraph = pc_model.predict(self._data)

        # Store the weighted adjacency matrix
        self.inferred_adjacency_mat = nx.to_numpy_array(self.inferred_DiGraph)

        # Display the matrix
        if self._verbose:
            logger.info(f'The inferred weighted adjacency matrix is: \n {self.inferred_adjacency_mat}')


class GIESAlg(InferenceAlgorithm):
    """
    Wrapper implementation for the GIES algorithm.

    The class is built upon the cdt.causality.graph.GIES library.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize a GIESAlg class.
        """

        super().__init__(verbose=verbose)

    def infer_causal_structure(self, start_graph: DiGraph = None):
        """
        Build a direct lingam model and infer the underlying structure.

        The structure is then stored in the class attribute 'inferred_adjacency_mat' for ease
        of access in future.
        """

        # Build and fit a direct lingam model
        gies_model = GIES()

        if start_graph is None:
            self.inferred_DiGraph = gies_model.predict(self._data)
        else:
            self.inferred_DiGraph = gies_model.predict(self._data, start_graph)

        # Store the weighted adjacency matrix
        self.inferred_adjacency_mat = nx.to_numpy_array(self.inferred_DiGraph)

        # Display the matrix
        if self._verbose:
            logger.info(f'The inferred weighted adjacency matrix is: \n {self.inferred_adjacency_mat}')


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
    """

    def __init__(self):
        """
        Initialize a ENCO Algorithm class.

        Note: Before running this function, you have to make sure that conda environment related to
        DSDI algorithm is up and running (named enco). For this purpose, follow the guidelines
        in the README file of the repository.

        """

        super().__init__()

    def build_global_dataset(self, obs_data_size: int, int_data_size: int, num_vars: int,
                             graph_type: str, seed: int = 0, num_categs: int = 10):
        """
        The function builds a graph and an external dataset using soft intervention and
        online sampling from the respective graph.
        """

        self._graph = generate_categorical_graph(num_vars=num_vars,
                                                 min_categs=num_categs,
                                                 max_categs=num_categs,
                                                 use_nn=True,
                                                 graph_func=get_graph_func(graph_type),
                                                 seed=seed)
        logger.info(f'Graph is built with the provided information: \n {graph}')

        self.original_adjacency_mat = graph.adj_matrix
        logger.info(f'Global dataset adjacency matrix: \n {adj_matrix}')

        self._data = graph.sample(batch_size=obs_data_size, as_array=True)
        logger.info(f'Shape of observational data: {self._data.shape}')

        self._data_int = self._sample_int_data(self, int_data_size)
        logger.info(f'Shape of interventional data: {self._data_int.shape}')

        self._global_dataset_dag = CausalDAGDataset(self.original_adjacency_mat,
                                                    self._data, self._data_int)

    def _sample_int_data(self, int_data_size: int):
        """
        Build an interventional dataset based on the provided parameters.
        """
        data_int: np.ndarray = None

        for var_idx in range(len(self._graph.variables)):

            # Select variable to intervene on
            var = self._graph.variables[var_idx]

            # Soft, perfect intervention => replace p(X_n) by random categorical
            # Scale is set to 0.0, which represents a uniform distribution.
            int_dist = _random_categ(size=(var.prob_dist.num_categs,), scale=0.0, axis=-1)

            # Sample from interventional distribution
            value = np.random.multinomial(n=1, pvals=int_dist,
                                        size=(int_data_size // len(self._graph.variables),))
            value = np.argmax(value, axis=-1)

            intervention_dict = {var.name: value}
            int_sample = graph.sample(interventions=intervention_dict,
                                    batch_size=(int_data_size // len(self._graph.variables)),
                                    as_array=True)

            data_int = np.array([int_sample]) if data_int is None \
                                              else np.append(data_int,
                                                              np.array([int_sample]),
                                                              axis=0)

    def load_local_dataset(self, accessible_data: List[str] or float,
                           assignment_type: str = 'observation_assignment',
                           dataset_name: str = "sachs", import_from_directory: bool = False):
        """
        Note: This is currently disabled in this class since the local dataset is distributed online,
        rather than how it is handled in other classes by distribution of a DataFrame and CSV files.

        Should be implemented upon usage of external datasets.
        """

        pass

    def _build_local_dataset(self, accessible_percentage: int, num_clients: int, client_id: int):
        """
        Build the local dataset for an specific client.
        """

        if len(self._data) == 0 or len(self._data_int) == 0:
            logger.error(f'Initialize a global dataset first!')



    def infer_causal_structure(self, dataset_dag: CausalDAGDataset, accessible_percentage: int = 100,
                               num_clients: int = 5, client_id: int = 0, round_id: int = 0,
                               experiment_id: int = 0, num_epochs: int = 10,
                               gamma_belief: str or None = None):
        """
        This function calls an inference algorithm using ENCO core functions and class, given a dataset_dag.

        The parameters may be set here, or leave them to the default values instead.
        For more information, refer to ENCO github page.
        """

        dataset_dag = self._build_local_dataset()

        enco_module = ENCO(graph=dataset_dag)
        if torch.cuda.is_available():
            logger.info('Found Cuda device!')
            enco_module.to(torch.device('cuda:0'))

        self.predicted_adj_matrix = enco_module.discover_graph(num_epochs=10)



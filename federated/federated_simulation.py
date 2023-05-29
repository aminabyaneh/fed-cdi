"""
    File name: federated_simulator.py
    Author: Amin Abyaneh
    Email: aminabyaneh@gmail.com
    Date created: 15/04/2021
    Python Version: 3.8
    Description: Design a simulated federated process based on ENCO as local learning method.
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

import logging
import os, sys
import pickle
import torch
import numpy as np
import shutil

from typing import Dict, List

sys.path.append("../")
from federated.utils import calculate_metrics, find_shortest_distance_dict
from federated.logging_settings import logger
from federated.causal_learning import ENCOAlg
from causal_graphs.graph_definition import CausalDAGDataset
from causal_discovery.utils import find_best_acyclic_graph


class FederatedSimulator:
    """
    Design a simulation for learning causal graphs in federated setup.

    A simple experimental run could be as follows.

        $ interventions_dict = {0: [c for c in range(15)], 1: [c for c in range(15, 30)]}
        $ federated_model = FederatedSimulator(interventions_dict)
        $ federated_model.initialize_clients()
        $ federated_model.execute_simulation()

    """

    def __init__(self, accessible_interventions: Dict[int, List[int]],
                 num_rounds: int = 5, num_clients: int = 2, experiment_id: int = 0,
                 repeat_id: int = 0, output_dir: str = 'default_federated_experiment',
                 client_parallelism: bool = False, verbose: bool = False):
        """ Initialize a federated setup for simulation.

        Args:
            accessible_interventions(Dict[int, List[int]]): A dictionary containing the number of
                intervened variables for each client. The key is client id.
            num_rounds (int, optional): Total number of federated rounds. Defaults to 5.
            num_clients (int, optional): Number of clients collaborating in the simulation. Defaults to 2.
            experiment_id (int, optional): Unique id of this experiment. Defaults to 0.
            repeat_id (int, optional): Number of random seeds for each simulation. Defaults to 0.
            output_dir (str, optional): Directory for saving the results. Defaults to
                'default_federated_experiment'.

            client_parallelism (bool, optional): Set True if you have enough GPU to give each client one.
                Defaults to False.
            verbose (bool, optional): Set True to see more detailed output. Defaults to False.
        """

        self.__num_rounds = num_rounds
        assert num_rounds > 0, "Number of rounds should be at least 1."

        self.__num_clients = num_clients
        assert num_clients > 0, "Number of clients cannot be lower than 1."

        self.__experiment_id = experiment_id
        self.__repeat_id = repeat_id
        self.__output_dir = output_dir
        os.makedirs(self.__output_dir, exist_ok=True)

        self.__num_vars = 0
        self.__clients : List[ENCOAlg] = list()
        self.__client_parallelism = client_parallelism
        self.__interventions_dict = accessible_interventions
        assert len(self.__interventions_dict.keys()) == self.__num_clients, \
            "Insufficient accessible interventions info."

        if verbose: logger.setLevel(logging.DEBUG)
        if self.__client_parallelism:
            gpu_count = torch.cuda.device_count()
            assert gpu_count >= self.__num_clients, \
                f'{gpu_count} GPU(s) not enough to perform {self.__num_clients}-client parallelism'
            torch.multiprocessing.set_start_method('spawn', force=True)

        self.results: Dict[str, List] = dict()
        self.initialize_results_dict()

    def initialize_clients_data(self, graph_type: str = "chain", num_vars = 30,
                                accessible_data_percentage: int = 100,
                                obs_data_size: int = 20000, int_data_size: int = 2000,
                                edge_prob: float or None = None, seed: int = 0,
                                external_global_dataset: CausalDAGDataset = None):
        """ Initialize client and clients' data for the number of clients in the federated setup.

        Args:
            graph_type (str, optional): Type of the graph. Defaults to "chain".
            num_vars (int, optional): Size of the graph. Defaults to 30.
            accessible_data_percentage (int, optional): The amount of local dataset that clients can see.
                Defaults to 100.

            obs_data_size (int, optional): Global observational dataset size. Defaults to 100000.
            int_data_size (int, optional): Global interventional dataset size. Defaults to 20000.
            edge_prob (floatorNone, optional): Edge existence probability only for random graphs.
                Defaults to None.
            seed (int, optional): Define a random seed for the dataset and graph generation.
                Defaults to 0.
        """

        # handle real-world data as an externally initialized dataset
        if external_global_dataset is not None:
            global_dataset_dag = external_global_dataset
            self.__num_vars = external_global_dataset.adj_matrix.shape[0]
            logger.info(f'External dataset parsed.')

        else:
            self.__num_vars = num_vars
            global_dataset_dag = ENCOAlg.build_global_dataset(obs_data_size, int_data_size,
                                                            num_vars, graph_type, edge_prob=edge_prob,
                                                            seed=seed)

        for client_id in range(self.__num_clients):
            try:
                enco_module = ENCOAlg(client_id, global_dataset_dag, accessible_data_percentage,
                                      self.__num_clients, self.__interventions_dict[client_id])
            except ValueError:
                logger.error(f'Global dataset missing for client {client_id}!')
                return

            self.__clients.append(enco_module)

    def initialize_results_dict(self):
        """Initializes the dictionary containing final results and per-round results.
        """

        self.results['round_adjs'] = list()
        self.results['round_gammas'] = list()
        self.results['round_thetas'] = list()
        self.results['round_metrics'] = list()
        self.results['round_acycle_adjs'] = list()
        self.results['round_acycle_metrics'] = list()

        self.results.update({f'client_{client_id}_metrics_acycle': list() for client_id in range(self.__num_clients)})
        self.results.update({f'client_{client_id}_metrics': list() for client_id in range(self.__num_clients)})
        self.results.update({f'client_{client_id}_adjs': list() for client_id in range(self.__num_clients)})

    def execute_simulation(self, aggregation_method: str = "naive", num_epochs: int = 2,
                           **kwargs):
        """ Execute the simulation based on the pre-defined federated setup.

        Args:
            aggregation_method (str, optional): Type of aggregation. Right now "locality", "naive"
                are implemented. Defaults to "naive".
            num_epochs (int, optional): Number of epochs for the local learning method.
                Defaults to 2.
            kwargs (dict, optinal):
                Any other argument that should be passed to the aggregation function.
        """

        assert len(self.__clients), "Clients are not initialized."
        assert aggregation_method in ["naive", "locality"], "Aggregation method not yet defined."

        prior_gamma: np.ndarray = None
        prior_theta: np.ndarray = None

        """ Federated loop """
        for round_id in range(self.__num_rounds):
            logger.info(f'Initiating round {round_id} of federated setup')

            """ Inference stage"""
            self.infer_local_models(prior_gamma, prior_theta, num_epochs)

            """ Aggregation stage """
            agg_gamma, agg_theta = self.aggregate_clients_updates(aggregation_method, round_id, **kwargs)

            """ Store round results """
            self.update_results(agg_gamma, agg_theta)

            """ Incorporate beliefs"""
            prior_gamma, prior_theta = agg_gamma, agg_theta

        """ Save the final results """
        self.save_results()

        logger.info(f'Finishing experiment {self.__experiment_id}\n')

    def infer_local_models(self, prior_gamma: np.ndarray, prior_theta: np.ndarray, num_epochs):
        """Execute the local learning methods for all clients.

        Note: Higher levels of parallelism are possible by defining client_parallelism in the instantiation step.

        Args:
            prior_gamma (np.ndarray): Prior for edge existence matrix.
            prior_theta (np.ndarray): Prior for edge orientation matrix.
            num_epochs (int): Number of epochs for ENCO.
        """

        if self.__client_parallelism:
            setup_cache_path = os.path.join(self.__output_dir, '.mpcache', f'res-{self.__experiment_id}')
            os.makedirs(setup_cache_path, exist_ok=True)

            clients_processes = list()
            for client in self.__clients:
                gpu_name = f'cuda:{client.get_client_id()}'
                setup_cache_file = os.path.join(setup_cache_path, f'{id(client)}.pickle')
                client_process = torch.multiprocessing.Process(target=client.infer_causal_structure,
                                                               args=(prior_gamma, prior_theta, num_epochs, gpu_name,
                                                                     setup_cache_file,))
                clients_processes.append(client_process)

            for client_p in clients_processes: client_p.start()
            for client_p in clients_processes: client_p.join()

            for client in self.__clients:
                setup_cache_file = os.path.join(setup_cache_path, f'{id(client)}.pickle')
                client.retrieve_results(setup_cache_file)
        else:
            for client in self.__clients:
                    client.infer_causal_structure(prior_gamma, prior_theta, num_epochs)

    def aggregate_clients_updates(self, aggregation_method, round_id, **kwargs):
        """Perform aggregation step for all clients.

        Args:
            aggregation_method (str): Can be naive or locality aggregation so far.
            round_id (int): Current roung id.

        Returns:
            np.ndarray, np.ndarray: Aggregated gamma and theta matrices.
        """

        if aggregation_method == "naive":
            agg_gamma, agg_theta = self.naive_aggregation()
        if aggregation_method == "locality":
            agg_gamma, agg_theta = self.locality_aggregation(round_id=round_id, **kwargs)

        return agg_gamma, agg_theta

    def naive_aggregation(self):
        """ Naive aggregation based on simple averaging and size of local dataset.

        Returns:
            numpy.ndarray: Prior for edge existence probabilities.
            numpy.ndarray: Prior for edge orientation probabilites.
        """

        weights: np.ndarray = np.zeros(shape=(self.__num_vars, 1))
        accumulated_gamma_mat: np.ndarray = np.zeros(shape=(self.__num_vars, self.__num_vars))
        accumulated_theta_mat: np.ndarray = np.zeros(shape=(self.__num_vars, self.__num_vars))

        for client in self.__clients:
            weighted_gamma_mat = client.inferred_existence_mat * client.get_accessible_percentage()
            accumulated_gamma_mat += weighted_gamma_mat

            weighted_theta_mat = client.inferred_orientation_mat * client.get_accessible_percentage()
            accumulated_theta_mat += weighted_theta_mat

            weights += client.get_accessible_percentage()

        prior_gamma: np.ndarray = accumulated_gamma_mat / weights
        prior_theta: np.ndarray = accumulated_theta_mat / weights

        return prior_gamma, prior_theta

    def locality_aggregation(self, initial_mass: np.ndarray, alpha: float, beta: float = 0,
                             min_mass: float = 1.0, round_id: int = 0):
        """ Aggregation of adjacency matrices based on locality (proximity).

        Args:
            initial_mass (numpy.ndarray): The initial mass given for each client.
            alpha (float): The reduction rate for mass flow.
            beta (float, optional): Temperature parameter for softmax. Defaults to 0.
            min_mass (float, optional): The minimum mass if no interventional info is available for
                an edge.
            round_id (int, optional): The current round id, utilized in finding the last round's
                results. Defaults to 0, so round_id - 1 points to the last element in the list.

        Returns:
            numpy.ndarray: Prior for edge existence probabilities.
            numpy.ndarray: Prior for edge orientation probabilites.
        """

        reference_adj_mat_exists: bool = len(self.results['round_adjs']) > 0
        reference_adjacency_mat = np.zeros(shape=(self.__num_vars, self.__num_vars))

        if reference_adj_mat_exists:
            reference_adjacency_mat = self.results['round_adjs'][round_id - 1]
            logger.debug(f'Setting reference adj matrix to prior: \n {reference_adjacency_mat} \n')

        aggregated_gamma_mat = np.zeros(shape=(self.__num_vars, self.__num_vars))
        aggregated_theta_mat = np.zeros(shape=(self.__num_vars, self.__num_vars))
        sum_gamma_scores = np.zeros(shape=(self.__num_vars, self.__num_vars))
        sum_theta_scores = np.zeros(shape=(self.__num_vars, self.__num_vars))

        for client in self.__clients:
            client_score_mat = np.full((self.__num_vars, self.__num_vars), min_mass)

            if not reference_adj_mat_exists:
                reference_adjacency_mat = client.binary_adjacency_mat
                print(reference_adjacency_mat.shape)
                logger.debug(f'Setting reference adj matrix to clients local: \n {reference_adjacency_mat} \n')

            distance_to_intervened = {var_idx: find_shortest_distance_dict(var_idx, reference_adjacency_mat) \
                                      for var_idx in client.get_interventions_list()}
            logger.debug(f'Shortest distance {client.get_client_id()}: \n {distance_to_intervened}')

            for v_i, v_j in np.transpose(np.nonzero(reference_adjacency_mat)):
                min_dist_int = np.min([distance_to_intervened[int_var][v_i] for int_var in client.get_interventions_list()])
                propagated_mass = np.power(alpha, min_dist_int) * initial_mass[client.get_client_id()]
                client_score_mat[v_i][v_j] = np.max([propagated_mass, min_mass])
            logger.debug(f'Reliability scores for client {client.get_client_id()}: \n {client_score_mat}\n')

            aggregated_gamma_mat += client_score_mat * client.inferred_existence_mat
            sum_gamma_scores += client_score_mat

            logger.debug(f'Client {client.get_client_id()} orientation mat: \n {client.inferred_orientation_mat}')
            aggregated_theta_mat += client_score_mat * client.inferred_orientation_mat
            sum_theta_scores += client_score_mat

        prior_gamma = aggregated_gamma_mat / sum_gamma_scores
        prior_theta = FederatedSimulator.adjust_theta(aggregated_theta_mat / sum_theta_scores)

        logger.debug(f'Aggregated gamma matrix: \n {prior_gamma}')
        logger.debug(f'Aggregated theta matrix: \n {prior_theta}')
        logger.debug(f'Aggregated sum scores matrix: \n {sum_theta_scores}')

        return prior_gamma, prior_theta

    def update_results(self, prior_gamma: np.ndarray, prior_theta: np.ndarray):
        """ Update the results dictionary for each round.

        Args:
            prior_gamma (numpy.ndarray): Edge existence matrix at the end of the round.
            prior_theta (numpy.ndarray): Edge orientation matrix acquired at the end of the round.
        """

        self.results['round_gammas'].append(prior_gamma)
        self.results['round_thetas'].append(prior_theta)

        ground_truth_matrix = self.__clients[0].original_adjacency_mat
        round_discovered_matrix = FederatedSimulator.get_binary_adjacency_mat(prior_gamma, prior_theta)
        round_acyclic_matrix = FederatedSimulator.get_acyclic_adjacency_mat(prior_gamma, prior_theta)

        round_metrics = calculate_metrics(round_discovered_matrix, ground_truth_matrix)
        round_acycle_metrics = calculate_metrics(round_acyclic_matrix, ground_truth_matrix)

        self.results['round_adjs'].append(round_discovered_matrix)
        self.results['round_metrics'].append(round_metrics)
        self.results['round_acycle_adjs'].append(round_acyclic_matrix)
        self.results['round_acycle_metrics'].append(round_acycle_metrics)

        for client in self.__clients:
            self.results[f'client_{client.get_client_id()}_adjs'].append(client.binary_adjacency_mat)
            self.results[f'client_{client.get_client_id()}_metrics'].append(client.metrics_dict)
            self.results[f'client_{client.get_client_id()}_metrics_acycle'].append(client.metrics_dict_acycle)

        logger.info(f'End of the round results: \n {round_discovered_matrix} \n {round_metrics} \n')

    def save_results(self):
        """ Save the results dictionary as a pickle file.
        """
        file_dir = os.path.join(self.__output_dir, f'results_{self.__experiment_id}_{self.__repeat_id}.pickle')
        with open(file_dir, 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        cache_dir = os.path.join(self.__output_dir, '.mpcache')
        # shutil.rmtree(cache_dir)

    @staticmethod
    def get_binary_adjacency_mat(gamma: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """ Calculate the adjacency matrix based on gamma and theta matrices.

        Args:
            gamma (numpy.ndarray): Edge existence matrix.
            theta (numpy.ndarray): Edge orientation matrix.

        Returns:
            numpy.ndarray: Binary adjacency matrix.
        """

        return (((gamma > 0.0) * (theta > 0.0)) == 1).astype(int)

    @staticmethod
    def get_acyclic_adjacency_mat(gamma: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """ Calculate the adjacency matrix based on acyclicity constraint.

        Args:
            gamma (numpy.ndarray): Edge existence matrix.
            theta (numpy.ndarray): Edge orientation matrix.

        Returns:
            numpy.ndarray: Binary and acyclic adjacency matrix.
        """
        gamma_t = torch.from_numpy(gamma)
        theta_t = torch.from_numpy(theta)

        acycle_mat_tensor = find_best_acyclic_graph(gamma=torch.sigmoid(gamma_t),
                                                    theta=torch.sigmoid(theta_t))

        return acycle_mat_tensor.numpy()

    @staticmethod
    def adjust_theta(prior_theta):
        """ Make the orientation matrix comply with ENCO rule of e_i,j + e_j,i = 1.

        Args:
            prior_theta (numpy.ndarray): Aggregated theta matrix.

        Returns:
            numpy.ndarray: Adjusted edge orientation matrix.
        """

        error_mat = prior_theta + prior_theta.T
        for v_i, v_j in np.transpose(np.nonzero(error_mat)):
            prob = np.max([np.abs(prior_theta[v_i][v_j]), np.abs(prior_theta[v_j][v_i])])
            prior_theta[v_i][v_j] = prob if prior_theta[v_i][v_j] > 0 else -prob
            prior_theta[v_j][v_i] = prob if prior_theta[v_i][v_j] < 0 else -prob

        return prior_theta

if __name__ == '__main__':
    interventions_dict = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7, 8]}

    federated_model = FederatedSimulator(interventions_dict, num_clients=2, num_rounds=10, client_parallelism=True)
    federated_model.initialize_clients_data(num_vars=10, graph_type="full")
    federated_model.execute_simulation(aggregation_method="locality",
                                       initial_mass=np.array([16, 16]),
                                       alpha=0.2, beta=0.3, min_mass=0.1)


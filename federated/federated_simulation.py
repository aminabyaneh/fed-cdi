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

import os
import pickle
import numpy as np

from typing import Dict, List

from causal_learning import ENCOAlg
from distributed_network import Network
from logging_settings import logger
from utils import calculate_metrics, find_shortest_distance_dict


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
                 verbose: bool = False):
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
        self.__interventions_dict = accessible_interventions
        assert len(self.__interventions_dict.keys()) == self.__num_clients, \
            "Insufficient accessible interventions info."

        self.results = {client_id: list() for client_id in range(self.__num_clients)}
        self.results['priors'] = list()
        self.results['matrices'] = list()

    def initialize_clients_data(self, graph_type: str = "chain", num_vars = 30,
                                accessible_data_percentage: int = 100,
                                obs_data_size: int = 20000, int_data_size: int = 2000,
                                edge_prob: float or None = None, seed: int = 0):
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

        logger.info(f'Running experiment {self.__experiment_id}')
        assert len(self.__clients), "Clients are not initialized."
        assert aggregation_method in ["naive", "locality"], "Aggregation method not yet defined."

        prior_gamma: np.ndarray = None
        prior_theta: np.ndarray = None

        """ Federated loop """
        for round_id in range(self.__num_rounds):
            logger.info(f'Initiating round {round_id} of federated setup')

            """ Inference stage """
            for client in self.__clients:
                client.infer_causal_structure(prior_gamma, prior_theta, num_epochs)

            """ Aggregation stage """
            if aggregation_method == "naive":
                prior_gamma, prior_theta = self.naive_aggregation()
            if aggregation_method == "locality":
                prior_gamma, prior_theta = self.locality_aggregation(round_id=round_id, **kwargs)

            """ Store round results """
            self.update_results(prior_gamma, prior_theta)

        """ Save the final results """
        self.save_results()

        logger.info(f'Finishing experiment {self.__experiment_id}\n')

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
        """ Aggregation of adjacency matrices based on locality.

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

        reference_adj_mat_exists: bool = len(self.results['matrices']) > 0
        reference_adjacency_mat = np.zeros(shape=(self.__num_vars, self.__num_vars))

        if reference_adj_mat_exists:
            reference_adjacency_mat = self.results['matrices'][round_id - 1][self.__num_clients - 1]
            logger.debug(f'Setting reference adj matrix to prior: \n {reference_adjacency_mat} \n')

        aggregated_gamma_mat = np.zeros(shape=(self.__num_vars, self.__num_vars))
        aggregated_theta_mat = np.zeros(shape=(self.__num_vars, self.__num_vars))
        sum_gamma_scores = np.zeros(shape=(self.__num_vars, self.__num_vars))
        sum_theta_scores = np.zeros(shape=(self.__num_vars, self.__num_vars))

        for client in self.__clients:
            client_score_mat = np.full((self.__num_vars, self.__num_vars), min_mass)

            if not reference_adj_mat_exists:
                reference_adjacency_mat = client.binary_adjacency_mat
                logger.debug(f'Setting reference adj matrix to clients local: \n {reference_adjacency_mat} \n')

            distance_to_intervened = {var_idx: find_shortest_distance_dict(var_idx, reference_adjacency_mat) \
                                      for var_idx in client.get_interventions_list()}
            logger.debug(f'Shortest distance {client.get_client_id()}: \n {distance_to_intervened}')

            for v_i, v_j in np.transpose(np.nonzero(reference_adjacency_mat)):
                min_dist_int = np.min([distance_to_intervened[int_var][v_i] for int_var in client.get_interventions_list()])
                logger.info(f'Min distance ({v_i},{v_j}): {min_dist_int}')
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

        ground_truth_matrix = self.__clients[0].original_adjacency_mat
        round_discovered_matrix = FederatedSimulator.get_binary_adjacency_mat(prior_gamma, prior_theta)
        round_metrics = calculate_metrics(round_discovered_matrix, ground_truth_matrix)
        self.results['priors'].append(round_metrics)

        clients_adjs = [client.binary_adjacency_mat for client in self.__clients]
        clients_adjs.append(round_discovered_matrix)
        self.results['matrices'].append(clients_adjs)

        for client in self.__clients:
            self.results[client.get_client_id()].append(client.metrics_dict)

        logger.info(f'End of the round results: \n {round_discovered_matrix} \n {round_metrics} \n')

    def save_results(self):
        """ Save the results dictionary as a pickle file.
        """
        file_dir = os.path.join(self.__output_dir, f'results_{self.__experiment_id}_{self.__repeat_id}.pickle')
        with open(file_dir, 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    federated_model = FederatedSimulator(interventions_dict, num_clients=2, num_rounds=10)
    federated_model.initialize_clients_data(num_vars=10, graph_type="full")
    federated_model.execute_simulation(aggregation_method="locality",
                                       initial_mass=np.array([16, 16]),
                                       alpha=0.2, beta=0.3, min_mass=0.1)


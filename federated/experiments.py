"""
    File name: experiments.py
    Author: Amin Abyaneh
    Email: aminabyaneh@gmail.com
    Date created: 15/04/2021
    Python Version: 3.8
    Description: Testing scenarios are implemented here.
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
import torch

from typing import List, Tuple
from torch.cuda import device_count

from causal_learning import ENCOAlg
from distributed_network import Network
from logging_settings import logger

from utils import generate_accessible_percentages


class Experiments:
    """
    Implementation of all the test functions.
    """

    def __init__(self):
        pass

    @staticmethod
    def test_network_broadcast():
        """
        Execution of broadcast function related to the distributed network module.
        """
        logger.info('\nEXPERIMENT STARTED: Network Broadcast')

        net = Network(network_size=5)
        net.execute_all_process(run_function=net.run_broadcast)

        logger.info('\nEXPERIMENT CONCLUDED: Network Broadcast\n')

    @staticmethod
    def test_network_send_recv():
        """
        Execution of send and receive functions related to the distributed network module.
        """
        logger.info('\nEXPERIMENT STARTED: Network Send/Recv')

        net = Network(network_size=5)
        net.execute_all_process(run_function=net.run_send_recv)

        logger.info('\n EXPERIMENT CONCLUDED: Network Send/Recv\n')

    @staticmethod
    def enco_federated(num_rounds: int = 5, num_clients: int = 5, experiment_id: int = 0,
                       folder_name: str = 'tests', accessible_data_range: Tuple = (40, 60),
                       obs_data_size: int = 100000, int_data_size: int = 20000, num_epochs: int = 2,
                       num_vars = 20, graph_type: str = "full", batch_size: int = 64):

        logger.info(f'EXPERIMENT {experiment_id} STARTED: ENCO Federated\n')
        logger.info(f'Found {device_count()} GPU devices')

        accessible_percentages_dict = generate_accessible_percentages(num_clients,
                                                                      accessible_data_range[0],
                                                                      accessible_data_range[1])
        # Dataset initialization
        clients: List[ENCOAlg] = list()
        for client_id in range(num_clients):
            if client_id == 0:
                # Generate a global dataset from scratch
                enco_module = ENCOAlg(client_id, accessible_percentages_dict[client_id],
                                      obs_data_size, int_data_size,
                                      num_vars, num_clients, graph_type, batch_size=batch_size)
                clients.append(enco_module)
            else:
                # Load a pre-existing global dataset
                enco_module = ENCOAlg(client_id=client_id,
                                      accessible_percentage=accessible_percentages_dict[client_id],
                                      num_clients=num_clients,
                                      external_dataset_dag=clients[0].global_dataset_dag,
                                      batch_size=batch_size)
                clients.append(enco_module)

        prior_gamma: np.ndarray = None
        prior_theta: np.ndarray = None

        results_dict = {client.get_client_id(): list() for client in clients}
        results_dict['priors'] = list()

        for round_id in range(num_rounds):
            logger.info(f'Initiating round {round_id}')

            # Inference stage
            for client in clients:
                client.infer_causal_structure(prior_gamma, prior_theta, num_epochs,
                                              round_id, experiment_id)

            # Aggregation stage gamma and theta
            accumulated_gamma_mat: np.ndarray = None
            accumulated_theta_mat: np.ndarray = None
            weights: int = 0

            for client in clients:
                results_dict[client.get_client_id()].append(client.metrics_dict)

                weighted_gamma_mat = client.inferred_adjacency_mat * client.get_accessible_percentage()
                accumulated_gamma_mat = weighted_gamma_mat if accumulated_gamma_mat is None \
                                                           else (accumulated_gamma_mat + weighted_gamma_mat)

                weighted_theta_mat = client.inferred_orientation_mat * client.get_accessible_percentage()
                accumulated_theta_mat = weighted_theta_mat if accumulated_theta_mat is None \
                                                           else (accumulated_theta_mat + weighted_theta_mat)

                weights += client.get_accessible_percentage()

            logger.info(f'Aggregation Result: \nWeights = {weights} '
                        f'\nAccumulation Matrix Gamma = \n{accumulated_gamma_mat}'
                        f'\nAccumulation Matrix Theta = \n{accumulated_theta_mat}')

            prior_gamma = accumulated_gamma_mat / weights
            prior_theta = accumulated_theta_mat / weights

            results_dict['priors'].append(clients[0].prior_metrics_dict)
            logger.info(f'End of the round results: {clients[0].prior_metrics_dict}')

        # Save the results dictionary
        save_dir = os.path.join(os.pardir, 'data', folder_name)
        os.makedirs(save_dir, exist_ok=True)

        file_dir = os.path.join(save_dir, f'results_{experiment_id}.pickle')
        with open(file_dir, 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        torch.cuda.empty_cache()
        logger.info(f'EXPERIMENT {experiment_id} CONCLUDED: ENCO Federated\n')


if __name__ == '__main__':

    Experiments.enco_federated()

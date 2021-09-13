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
import numpy as np

from typing import List, Tuple
from numpy.lib.function_base import hamming
from torch.cuda import device_count
from torch.multiprocessing import Process, set_start_method


from causal_learning import DSDIAlg, ENCOAlg
from distributed_network import Network
from logging_settings import logger

from utils import generate_accessible_percentages
from utils import save_data_object
from utils import retrieve_dsdi_stored_data

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
    def dsdi_federated(experiment_id: int = 0, number_of_rounds: int = 10,
                                  number_of_clients: int = 5, accessible_segment=(100, 100),
                                  graph_structure: str = 'chain3',
                                  store_folder: str = 'default_experiments',
                                  num_epochs: int = 50,
                                  train_functional: int = 5000,
                                  epi_size: int = 10,
                                  dpe: int = 10,
                                  ipd: int = 100):
        """
        Emulating a federated setup using the DSDIAlg class.

        Args:
            train_functional (int): Size of the general dataset. Defaults to 5000.
            store_folder (str): Indicating a folder to store the results coming from clients at the end of each round.
            graph_structure (str): Determine the graph structure for the DSDI algorithm.
            number_of_clients (int): Determine the number of clients during the simulation.
            accessible_segment (Tuple): A tuple indicating two percentages for accessible data.
            number_of_rounds (int): The number of federated back-and-forth rounds.
            experiment_id (int): A way to distinct between various output files.
            Defaults to 0 to replace output results.
        """

        logger.info(f'\nEXPERIMENT {experiment_id} STARTED: DSDI Federated\n')

        # Generate accessible parts of dataset
        accessible_percentages_dict = generate_accessible_percentages(number_of_clients, accessible_segment[0],
                                                                      accessible_segment[1])

        for round_id in range(number_of_rounds):
            logger.info(f'Initiating round {round_id}')

            # Accessing prior info
            prior_dir = os.path.join('data', 'priors', f'{store_folder}')

            # LOCAL LEARNING STEP
            def run_local_learning(c_id, r_id):
                logger.info(f'Client {c_id} started the process')

                # Build a DSDI model
                client_model = DSDIAlg()

                # Infer the underlying structure and weights
                if r_id == 0:
                    # Clients must compute without a prior belief for the first round
                    client_model.infer_causal_structure(accessible_percentage=int(accessible_percentages_dict[c_id]),
                                                        num_clients=number_of_clients,
                                                        client_id=c_id,
                                                        round_id=r_id,
                                                        experiment_id=experiment_id,
                                                        num_epochs=num_epochs,
                                                        train_functional=train_functional,
                                                        epi_size=epi_size,
                                                        dpe=dpe,
                                                        ipd=ipd,
                                                        graph=graph_structure,
                                                        store_folder=store_folder)

                if r_id != 0:
                    # Clients must compute with a prior belief after the first round
                    client_model.infer_causal_structure(accessible_percentage=int(accessible_percentages_dict[c_id]),
                                                        num_clients=number_of_clients,
                                                        client_id=c_id,
                                                        round_id=r_id,
                                                        experiment_id=experiment_id,
                                                        num_epochs=num_epochs,
                                                        train_functional=train_functional,
                                                        epi_size=epi_size,
                                                        dpe=dpe,
                                                        ipd=ipd,
                                                        gamma_belief=os.path.join(prior_dir, f'prior_info_{experiment_id}.npy'),
                                                        graph=graph_structure,
                                                        store_folder=store_folder)

            process_list = list()
            for client_id in range(1, number_of_clients + 1):
                process_list.append(Process(target=run_local_learning, args=(client_id, round_id,)))
                process_list[-1].start()

            for process in process_list:
                process.join()
            process_list.clear()

            # AGGREGATION STEP
            logger.info(f'Calculated matrices:\n')
            aggregated_adjacency_matrix: np.ndarray = None
            access_sum: int = 0

            data_dir = os.path.join('work', store_folder)
            for data in retrieve_dsdi_stored_data(data_dir, experiment_id, round_id):
                logger.info(f'Retrieved data: \n {data}')

                if aggregated_adjacency_matrix is None:
                    aggregated_adjacency_matrix = data[0] * data[1]
                else:
                    aggregated_adjacency_matrix += data[0] * data[1]

                access_sum += data[0]
            print('\n')

            logger.info(f'Voting result: {access_sum}, \n{aggregated_adjacency_matrix}')
            prior_info: np.ndarray = aggregated_adjacency_matrix / access_sum
            logger.info(f'Prior matrix: \n{prior_info}')

            os.makedirs(prior_dir, exist_ok=True)
            save_data_object(prior_info, f'prior_info_{experiment_id}', prior_dir)

        logger.info(f'\nEXPERIMENT {experiment_id} CONCLUDED: DSDI Federated \n')

    @staticmethod
    def enco_federated(num_rounds: int = 5, num_clients: int = 5, experiment_id: int = 0,
                       accessible_data_range: Tuple = (100, 100),
                       obs_data_size: int = 100000, int_data_size: int = 20000,
                       int_data_batches: int = 1, num_epochs: int = 2,
                       num_vars = 20, graph_type: str = "full"):

        logger.info(f'EXPERIMENT {experiment_id} STARTED: ENCO Federated\n')
        logger.info(f'Found {device_count()} GPU devices')

        try:
            set_start_method('spawn')
        except:
            logger.critical('Failed to set the start method')
            return

        accessible_percentages_dict = generate_accessible_percentages(num_clients,
                                                                      accessible_data_range[0],
                                                                      accessible_data_range[1])
        # Dataset initialization
        clients: List[ENCOAlg] = list()
        for client_id in range(num_clients):
            if client_id == 0:
                # Generate a global dataset from scratch
                enco_module = ENCOAlg(client_id, 100,
                                      obs_data_size, int_data_size,
                                      num_vars, num_clients, graph_type)
                clients.append(enco_module)
            else:
                # Load a pre-existing global dataset
                enco_module = ENCOAlg(client_id=client_id,
                                      accessible_percentage=accessible_percentages_dict[client_id],
                                      num_clients=num_clients,
                                      external_dataset_dag=clients[0].global_dataset_dag)
                clients.append(enco_module)

        prior_mat: np.ndarray = None
        results_dict = {client.get_client_id(): hamming_dist for client in clients}
        for round_id in range(num_rounds):
            logger.info(f'Initiating round {round_id}')

            # Inference stage
            process_list = list()
            for client in clients:
                process_list.append(Process(target=client.infer_causal_structure,
                                            args=(prior_mat, num_epochs, round_id, experiment_id)))
                process_list[-1].start()

            for process in process_list:
                process.join()
            process_list.clear()

            # Reload the results
            for client in clients:
                client.load_gamma()

            # Aggregation stage
            accumulated_adj_mat: np.ndarray = None
            weights: int = 0

            for client in clients:
                weighted_mat = client.inferred_adjacency_mat * client.get_accessible_percentage()
                accumulated_adj_mat = weighted_mat if accumulated_adj_mat is None \
                                                    else (accumulated_adj_mat + weighted_mat)

                weights += client.get_accessible_percentage()

            logger.info(f'Aggregation result: \nWeights = {weights} '
                        f'\nAccumulation_Matrix = \n{accumulated_adj_mat}')

            prior_mat = accumulated_adj_mat / weights

        logger.info(f'EXPERIMENT {experiment_id} CONCLUDED: ENCO Federated\n')


if __name__ == '__main__':
    Experiments.enco_federated()

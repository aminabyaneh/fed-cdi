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
import glob
import logging
import os
import time

import numpy as np

from causal_learning import InferenceAlgorithm, DSDIAlg
from distributed_network import Network
from logging_settings import logger
from utils import evaluate_inferred_matrix, resume_dsdi_experiments
from utils import generate_accessible_percentages
from utils import save_data_object
from utils import retrieve_dsdi_stored_data
from typing import Callable, List
from multiprocessing import Pool, Process

"""
Prerequisites (related to CDT): 

    1. Install R.
    2. Go into R terminal and type: install.packages("BiocManager")
    3.1 In the shell: 
            sudo apt install libxml2-dev
            sudo apt install libgsl-dev

    3.2 Also type this in R terminal: 
            BiocManager::install(c("CAM", "SID", "bnlearn", "pcalg", "kpcalg", "D2C", "devtools", 
                                   "momentchi2", "MASS", "gsl"))
            
            Note: you might have to install each separately and in different order if one does not 
            work properly. 

    4. In your shell:

            sudo apt-get -y build-dep libcurl4-gnutls-dev
            sudo apt-get -y install libcurl4-gnutls-dev

    5. Follow: https://github.com/Diviyan-Kalainathan/RCIT
"""


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
    def experiment_causal_algorithms(algorithm: Callable, experiment_id: int = 0, verbose: bool = True,
                                     show_graphs: bool = False):
        """
        Testing the results of causal inference algorithms.

        Args:
            algorithm (InferenceAlgorithm): Class of the algorithm to be selected for the experiment.
            The acceptable classes are available in causal_learning file.

            verbose (bool): Whether to show logs or not during the experiment.
            experiment_id (int): A way to distinct between various output files.
            Defaults to 0 to replace output results.

            show_graphs (bool): Whether to display the original causal graphs or not.
        """

        logger.info(f'\nEXPERIMENT {experiment_id} STARTED: {algorithm.__name__}\n')

        # Determine the number of clients
        number_of_clients = 50

        # Generate accessible parts of dataset
        accessible_percentages_dict = generate_accessible_percentages(number_of_clients, 5, 100)

        # Find a baseline result using hole dataset
        client_model = algorithm()
        client_model.load_local_dataset(accessible_data=100)

        # Infer the underlying structure and weights
        client_model.infer_causal_structure()
        baseline_adjacency_matrix = client_model.inferred_adjacency_mat
        logger.info(f'The baseline matrix is: \n {baseline_adjacency_matrix}')

        # Show the baseline graph
        if show_graphs:
            client_model.visualize_causal_graph(file_name='baseline')

        # Create a dictionary to store results
        warm_start_graph = None
        output_dictionary = dict()

        for client_id in range(1, number_of_clients + 1):

            # Build a lingam model and load dataset
            client_model = algorithm(verbose=False)
            client_model.load_local_dataset(accessible_data=accessible_percentages_dict[client_id])

            # Infer the underlying structure and weights
            if warm_start_graph is None:
                client_model.infer_causal_structure()
            else:
                logger.info('Warm up start initiated')
                client_model.infer_causal_structure(warm_start_graph)

            # Evaluate the result
            evaluation_result = evaluate_inferred_matrix(baseline_adjacency_matrix,
                                                         client_model.inferred_adjacency_mat)

            if verbose:
                logger.info(f'\n Client Id: {client_id} '
                            f'\t Dataset size: {client_model.get_observations_size()}'
                            f'\n Inferred matrix: \n {client_model.inferred_adjacency_mat} \n'
                            f' Euclidean Distance: {evaluation_result["ED"]} \n'
                            f' Precision-recall: {evaluation_result["PR"]} \n'
                            f' SID Score: {evaluation_result["SID"]} \n'
                            f' SHD Score: {evaluation_result["SHD"]} \n')

            # Show the acquired graph
            if show_graphs:
                client_model.visualize_causal_graph(file_name='client' + str(client_id))

            # Create a dictionary based on observation sizes and metrics
            output_dictionary[client_model.get_observations_size()] = evaluation_result
            save_data_object(output_dictionary, f'{algorithm.__name__}_Observation_Sweep_{str(experiment_id)}',
                             save_directory='output')

        logger.info(f'\nEXPERIMENT {experiment_id} CONCLUDED: {algorithm.__name__} \n')

    @staticmethod
    def experiment_dsdi_federated(experiment_id: int = 0, number_of_rounds: int = 10,
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

        # Determine the number of clients
        number_of_clients = number_of_clients

        # Determine the number of rounds
        number_of_rounds = number_of_rounds

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
            logging.info(f'Calculated matrices:\n')
            aggregated_adjacency_matrix: np.ndarray = None
            access_sum: int = 0

            data_dir = os.path.join('work', store_folder)
            for data in retrieve_dsdi_stored_data(data_dir, experiment_id, round_id):
                print(data)

                if aggregated_adjacency_matrix is None:
                    aggregated_adjacency_matrix = data[0] * data[1]
                else:
                    aggregated_adjacency_matrix += data[0] * data[1]

                access_sum += data[0]
            print('\n')

            logging.info(f'Voting result: {access_sum}, \n{aggregated_adjacency_matrix}')
            prior_info: np.ndarray = aggregated_adjacency_matrix / access_sum
            logging.info(f'Prior matrix: \n{prior_info}')

            os.makedirs(prior_dir, exist_ok=True)
            save_data_object(prior_info, f'prior_info_{experiment_id}', prior_dir)

        logger.info(f'\nEXPERIMENT {experiment_id} CONCLUDED: DSDI Federated \n')


if __name__ == '__main__':

    logger.info('Starting the experiment sequence\n')

    # Change to CausalLearningFederated sub-module
    if os.path.basename(os.getcwd()) != 'CausalLearningFederated':
        os.chdir(os.pardir)

    number_of_experiments = 1

    init_time = time.time()
    for ex in range(number_of_experiments):
        Experiments.experiment_dsdi_federated(experiment_id=ex, number_of_rounds=2,
                                              number_of_clients=10,
                                              graph_structure='chain10', store_folder='test')

    logger.info('All the experiments have been executed')

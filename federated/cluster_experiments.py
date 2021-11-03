"""
    File name: cluster_experiments.py
    Author: Amin Abyaneh
    Email: aminabyaneh@gmail.com
    Date created: 15/07/2021
    Python Version: 3.8
    Description: Experiments for the Tuebingen cluster.
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

import sys
import numpy as np

from federated_simulation import FederatedSimulator
from logging_settings import logger
from utils import split_variables_set


def parallel_experiments_enco_rnd():
    """ A method to handle parallel MPI cluster experiments.
    """
    process = int(str(sys.argv[1]))
    logger.info(f'Starting the experiment sequence for process {process}\n')

    # Id
    experiment_id = process
    repeat_count = 5

    # Graph
    graph_type = "random"
    num_vars = 50
    edge_probs = [0.01, 0.02, 0.04, 0.8, 0.16, 0.32, 0.64]

    # Federated
    num_rounds = 10
    num_clients = 10
    c_num = 1

    obs_data_sizes = [500 * num_clients, 1000 * num_clients, 1500 * num_clients,
                      2000 * num_clients, 2500 * num_clients, 3000 * num_clients,
                      3500 * num_clients, 4000 * num_clients, 5000 * num_clients,
                      6000 * num_clients, 7000 * num_clients, 8000 * num_clients]

    int_data_sizes = [32 * (p * num_vars) * num_clients for p in range(1, 12 + 1)]

    interventions_dict = {cid: [v for v in range(num_vars)] for cid in range(num_clients)}

    # TODO: Not ready to run, adapt first!
    for edge_prob in edge_probs:
        folder_name = f'Graph{c_num}-{graph_type}{edge_prob}-{num_vars}' if c_num == num_clients \
                      else f'Graph{c_num}x-{graph_type}{edge_prob}-{num_vars}'
        for seed in range(repeat_count):
            federated_model = FederatedSimulator(interventions_dict, num_clients=num_clients[experiment_id],
                                                 num_rounds=num_rounds, experiment_id=experiment_id,
                                                 repeat_id=seed, output_dir=folder_name)
            federated_model.initialize_clients_data(num_vars=num_vars, graph_type=graph_type,
                                                    obs_data_size=obs_data_sizes[experiment_id],
                                                    int_data_size=int_data_sizes[experiment_id],
                                                    edge_prob= edge_prob, seed=seed)
            federated_model.execute_simulation(aggregation_method="naive")

    logger.info(f'Ending the experiment sequence for process {process}\n')


def parallel_experiments_enco_str():
    """ A method to handle parallel MPI cluster experiments.
    """
    process = int(str(sys.argv[1]))
    logger.info(f'Starting the experiment sequence for process {process}\n')

    # Id
    experiment_id = process
    repeat_count = 5

    # Federated
    num_rounds = 10
    num_clients = [10, 1, 10]
    c_nums = [10, 1, 1]

    # Graph
    num_vars = 50
    graph_types = ['chain', 'jungle', 'full']
    interventions_dict = {cid: [v for v in range(num_vars)] for cid in range(num_clients)}

    # TODO: Not ready to run, adapt first!
    for c_num, num_client in zip(c_nums, num_clients):
        for graph_type in graph_types:
            obs_data_sizes, int_data_sizes = get_datasets_size_naive(graph_type, num_client, num_vars)
            folder_name = f'Graph{c_num}-{graph_type}-{num_vars}' if c_num == num_client \
                          else f'Graph{c_num}x-{graph_type}-{num_vars}'

            logger.info(f'Starting experiments for {c_num} client {num_client} setting')
            for seed in range(repeat_count):
                federated_model = FederatedSimulator(interventions_dict, num_clients=num_clients[experiment_id],
                                                     num_rounds=num_rounds, experiment_id=experiment_id,
                                                     repeat_id=seed, output_dir=folder_name)
                federated_model.initialize_clients_data(num_vars=num_vars, graph_type=graph_type,
                                                        obs_data_size=obs_data_sizes[experiment_id],
                                                        int_data_size=int_data_sizes[experiment_id], seed=seed)
                federated_model.execute_simulation(aggregation_method="naive")

    logger.info(f'Ending the experiment sequence for process {process}\n')


def parallel_experiments_alpha_search():
    """ A method to handle parallel MPI cluster experiments.
    """
    process = int(str(sys.argv[1]))
    logger.info(f'Starting the experiment sequence for process {process}\n')

    # Id
    experiment_id = process

    # Graph
    graph_types = ["chain", "jungle", "collider", "full", "bidiag"]
    interventions_dict = {0: [v for v in range(10)],
                          1: [v for v in range(10, 20)]}
    num_vars = 20

    # Federated
    num_rounds = 10
    num_clients = [2, 2, 2, 2, 2, 2, 2, 2]
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
    initial_masses = [1, 5, 10, 15, 20, 25]
    repeat = 0

    for graph_type in graph_types:
        for initial_mass in initial_masses:
            folder_name = f'AlphaSearch-{graph_type}-{num_vars}-{initial_mass}'
            obs_data_size, int_data_size = get_datasets_size_locality(graph_type, num_clients[experiment_id], num_vars)
            federated_model = FederatedSimulator(interventions_dict, num_clients=num_clients[experiment_id],
                                                num_rounds=num_rounds, experiment_id=experiment_id,
                                                repeat_id=repeat, output_dir=folder_name)

            federated_model.initialize_clients_data(num_vars=num_vars, graph_type=graph_type, obs_data_size=obs_data_size,
                                                int_data_size=int_data_size)

            federated_model.execute_simulation(aggregation_method="locality",
                                                initial_mass=np.array([initial_mass, initial_mass]),
                                                alpha=alphas[experiment_id], beta=0.3, min_mass=1)

    logger.info(f'Ending the experiment sequence for process {process}\n')


def parallel_experiments_toy_str():
    """ A method to handle parallel MPI cluster experiments.
    """
    process = int(str(sys.argv[1]))
    logger.info(f'Starting the experiment sequence for process {process}\n')

    # Id
    experiment_id = process
    specifier = "unbl"

    # Graph
    graph_types = ["jungle", "collider", "chain", "full", "bidiag"]
    num_vars = 20
    accessible_percentages = [30, 70]

    # Federated
    num_rounds = 10
    num_clients = [1, 1, 2]
    repeat = 10

    for graph_type in graph_types:
        obs_data_size, int_data_size = get_datasets_size_locality(graph_type, num_clients[experiment_id],
                                                                  num_vars)
        for seed in range(repeat):
            folder_name = f'ToySetup-{graph_type}-{num_vars}-{specifier}'

            splits = split_variables_set(num_vars, accessible_percentages, seed)
            interventions_dict = [{0: splits[0]}, {0: splits[1]}, # Single client setup
                                  {0: splits[0], 1: splits[1]}] # Federated collaboration

            federated_model = FederatedSimulator(interventions_dict[experiment_id],
                                                 num_clients=num_clients[experiment_id],
                                                 num_rounds=num_rounds, experiment_id=experiment_id,
                                                 repeat_id=seed, output_dir=folder_name)

            federated_model.initialize_clients_data(num_vars=num_vars, graph_type=graph_type,
                                                    obs_data_size=obs_data_size,
                                                    int_data_size=int_data_size, seed=seed)

            federated_model.execute_simulation(aggregation_method="locality",
                                               initial_mass=np.array([16, 16]),
                                               alpha=1, beta=0.3, min_mass=1)

    logger.info(f'Ending the experiment sequence for process {process}\n')


def parallel_experiments_toy_rnd():
    """ A method to handle parallel MPI cluster experiments.
    """
    process = int(str(sys.argv[1]))
    logger.info(f'Starting the experiment sequence for process {process}\n')

    # Id
    experiment_id = process
    specifier = "unbl"
    accessible_percentages = [30, 70]

    # Graph
    graph_type = "random"
    edge_probs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    num_vars = 20

    # Federated
    num_rounds = 10
    num_clients = [1, 1, 2]
    repeat = 10

    for edge_prob in edge_probs:
        obs_data_size, int_data_size = get_datasets_size_locality(graph_type, num_clients[experiment_id],
                                                                  num_vars, edge_prob)
        for seed in range(repeat):
            folder_name = f'ToySetup-{graph_type}-{num_vars}-{edge_prob}-{specifier}'

            splits = split_variables_set(num_vars, accessible_percentages, seed)
            interventions_dict = [{0: splits[0]}, {0: splits[1]}, # Single client setup
                                  {0: splits[0], 1: splits[1]}] # Federated collaboration

            federated_model = FederatedSimulator(interventions_dict[experiment_id],
                                                 num_clients=num_clients[experiment_id],
                                                 num_rounds=num_rounds, experiment_id=experiment_id,
                                                 repeat_id=seed, output_dir=folder_name)

            federated_model.initialize_clients_data(num_vars=num_vars, graph_type=graph_type,
                                                    obs_data_size=obs_data_size,
                                                    int_data_size=int_data_size,
                                                    edge_prob=edge_prob, seed=seed)

            federated_model.execute_simulation(aggregation_method="locality",
                                               initial_mass=np.array([16, 16]),
                                               alpha=1, beta=0.3, min_mass=1)

    logger.info(f'Ending the experiment sequence for process {process}\n')


def get_datasets_size_locality(graph_type: str, num_clients: int, num_vars: int, edge_prob: float = 0.0):

    """ Chain, Jungle, Collider, and Bidiag graphs sample sizes """
    if graph_type == 'chain' or graph_type == 'jungle' or graph_type == 'collider' or graph_type == 'bidiag':
        obs_data_sizes = 5000 * num_clients
        int_data_sizes = 32 * (2 * num_vars) * num_clients

    """ Full graph sample sizes """
    if graph_type == 'full':
        obs_data_sizes = 10000 * num_clients
        int_data_sizes = 32 * (4 * num_vars) * num_clients

    """ Random graphs sample sizes """
    if graph_type == 'random':
        obs_data_sizes = edge_prob * 10000 * num_clients
        int_data_sizes = 32 * (2 * num_vars) * num_clients

    return obs_data_sizes, int_data_sizes


def get_datasets_size_naive(graph_type: str, num_clients: int, num_vars: int):
    """ Chain graph sample sizes """
    if graph_type == 'chain':
        obs_data_sizes = [260 * num_clients, 280 * num_clients, 300 * num_clients,
                          320 * num_clients, 350 * num_clients, 370 * num_clients,
                          390 * num_clients, 410 * num_clients, 430 * num_clients,
                          450 * num_clients, 335 * num_clients, 500 * num_clients,]
        int_data_sizes = [32 * (p * num_vars) * num_clients for p in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    """ Jungle graph sample sizes """
    if graph_type == 'jungle':
        obs_data_sizes = [450 * num_clients, 550 * num_clients, 600 * num_clients,
                          650 * num_clients, 700 * num_clients, 750 * num_clients,
                          800 * num_clients, 850 * num_clients, 900 * num_clients,
                          950 * num_clients, 1000 * num_clients, 1100 * num_clients,]
        int_data_sizes = [32 * (p * num_vars) * num_clients for p in [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    """ Collider graph sample sizes """
    if graph_type == 'collider':
        obs_data_sizes = [500 * num_clients, 1000 * num_clients, 1500 * num_clients,
                          2000 * num_clients, 2500 * num_clients, 3000 * num_clients,
                          3500 * num_clients, 4000 * num_clients, 5000 * num_clients,
                          6000 * num_clients, 7000 * num_clients, 8000 * num_clients]
        int_data_sizes = [32 * (p * num_vars) * num_clients for p in range(1, 12 + 1)]

    """ Full graph sample sizes """
    if graph_type == 'full':
        obs_data_sizes = [50000 * num_clients, 100000 * num_clients, 150000 * num_clients,
                          200000 * num_clients, 250000 * num_clients, 300000 * num_clients,
                          350000 * num_clients, 400000 * num_clients, 500000 * num_clients,
                          600000 * num_clients, 700000 * num_clients, 800000 * num_clients]
        int_data_sizes = [32 * (p * num_vars) * num_clients * 100 for p in range(1, 12 + 1)]

    return obs_data_sizes, int_data_sizes


if __name__ == '__main__':
    parallel_experiments_alpha_search()


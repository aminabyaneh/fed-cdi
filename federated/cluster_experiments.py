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
from experiments import Experiments
from logging_settings import logger

def parallel_experiments_enco():
    """ A method to handle parallel MPI cluster experiments.
    """
    process = int(str(sys.argv[1]))
    logger.info(f'Starting the experiment sequence for process {process}\n')

    """ Configurations """

    # Id
    experiment_id = process

    # Graph
    graph_type = "full"
    num_vars = 40

    # Federated
    num_rounds = 10
    num_clients = 10
    c_num = 10

    accessible_data = (100, 100)
    obs_data_sizes = [500 * num_clients, 1000 * num_clients, 1500 * num_clients,
                      2000 * num_clients, 2500 * num_clients, 3000 * num_clients,
                      3500 * num_clients, 4000 * num_clients, 5000 * num_clients,
                      6000 * num_clients, 7000 * num_clients, 8000 * num_clients]

    int_data_sizes = [32 * (p * num_vars) * num_clients for p in range(1, 12 + 1)]

    num_epochs = 2
    folder_name = f'Graph{c_num}-{graph_type}-{num_vars}' if c_num == num_clients else f'Graph{c_num}-{graph_type}-{num_vars}-all'

    Experiments.enco_federated(num_rounds, c_num, experiment_id,
                               folder_name,
                               accessible_data, obs_data_sizes[experiment_id],
                               int_data_sizes[experiment_id], num_epochs, num_vars, graph_type)

    logger.info(f'Ending the experiment sequence for process {process}\n')

def parallel_experiments_enco_rnd():
    """ A method to handle parallel MPI cluster experiments.
    """
    process = int(str(sys.argv[1]))
    logger.info(f'Starting the experiment sequence for process {process}\n')

    """ Configurations """

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

    accessible_data = (100, 100)
    obs_data_sizes = [500 * num_clients, 1000 * num_clients, 1500 * num_clients,
                      2000 * num_clients, 2500 * num_clients, 3000 * num_clients,
                      3500 * num_clients, 4000 * num_clients, 5000 * num_clients,
                      6000 * num_clients, 7000 * num_clients, 8000 * num_clients]

    int_data_sizes = [32 * (p * num_vars) * num_clients for p in range(1, 12 + 1)]

    num_epochs = 2

    for edge_prob in edge_probs:
        folder_name = f'Graph{c_num}-{graph_type}{edge_prob}-{num_vars}' if c_num == num_clients else f'Graph{c_num}x-{graph_type}{edge_prob}-{num_vars}'
        for repeat_id in range(repeat_count):
            Experiments.enco_federated(num_rounds, c_num, experiment_id, repeat_id,
                                       folder_name,
                                       accessible_data, obs_data_sizes[experiment_id],
                                       int_data_sizes[experiment_id], num_epochs, num_vars,
                                       graph_type, edge_prob=edge_prob)

    logger.info(f'Ending the experiment sequence for process {process}\n')


def parallel_experiments_enco_str():
    """ A method to handle parallel MPI cluster experiments.
    """
    process = int(str(sys.argv[1]))
    logger.info(f'Starting the experiment sequence for process {process}\n')

    """ Configurations """

    # Id
    experiment_id = process
    repeat_count = 5

    # Graph
    # graph_types = ['chain', 'collider', 'jungle', 'full']
    graph_types = ['full']
    num_vars = 50

    # Federated
    num_rounds = 10
    num_clients = 10
    c_num = 10

    accessible_data = (100, 100)
    obs_data_sizes = [500 * num_clients, 1000 * num_clients, 1500 * num_clients,
                      2000 * num_clients, 2500 * num_clients, 3000 * num_clients,
                      3500 * num_clients, 4000 * num_clients, 5000 * num_clients,
                      6000 * num_clients, 7000 * num_clients, 8000 * num_clients]

    int_data_sizes = [32 * (p * num_vars) * num_clients for p in range(1, 12 + 1)]

    num_epochs = 2

    for graph_type in graph_types:
        logger.info(f'Starting experiments for {graph_type} graphs')
        folder_name = f'Graph{c_num}-{graph_type}-{num_vars}' if c_num == num_clients else f'Graph{c_num}x-{graph_type}-{num_vars}'
        for repeat_id in range(repeat_count):
            Experiments.enco_federated(num_rounds, c_num, experiment_id, repeat_id,
                                       folder_name,
                                       accessible_data, obs_data_sizes[experiment_id],
                                       int_data_sizes[experiment_id], num_epochs, num_vars,
                                       graph_type, edge_prob=1)

    logger.info(f'Ending the experiment sequence for process {process}\n')


if __name__ == '__main__':
    parallel_experiments_enco_str()

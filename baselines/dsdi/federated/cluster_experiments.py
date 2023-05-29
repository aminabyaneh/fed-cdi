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

import os
import sys

from logging_settings import logger
from experiments import Experiments


def dsdi_parallel_experiments_balanced_int(number_of_experiments=8):

    process = int(str(sys.argv[1]))

    # Configurations
    exp_id = process % 3
    seed = process // 3

    # Federated setup
    num_rounds = 10
    graph_types = ['bidiag10'] # ['full10', 'chain10', 'collider10', 'jungle10']
    num_clients = [2, 1, 1]
    clients_data_coeff = [2, 1, 2]

    # Data setup
    train_data = 9000 * clients_data_coeff[exp_id]
    int_data = 8 * clients_data_coeff[exp_id]

    logger.info(f'Starting the experiment sequence for {num_clients[exp_id]} clients...\n')

    # Assuming we're in federated folder
    os.chdir(os.pardir)
    logger.info(f'Current pwd: {os.getcwd()}')

    # Run the experiment
    for graph_type in graph_types:
        store_folder = f'balanced_{graph_type}_{num_clients[exp_id]}client_{clients_data_coeff[exp_id]}data'
        Experiments.experiment_dsdi_federated(experiment_id=seed,
                                              number_of_rounds=num_rounds,
                                              number_of_clients=num_clients[exp_id],
                                              accessible_segment=(100, 100),
                                              graph_structure=graph_type,
                                              store_folder=store_folder,
                                              train_functional=train_data,
                                              epi_size=int_data,
                                              seed=seed,
                                              num_epochs=10)

    logger.info(f'Ending the experiment sequence for process {process}\n')


if __name__ == '__main__':
    dsdi_parallel_experiments_balanced_int()

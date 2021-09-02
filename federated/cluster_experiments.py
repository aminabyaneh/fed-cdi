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


def parallel_experiment():

    process = int(str(sys.argv[1]))

    # Configurations
    number_of_experiments = 8
    
    experiment_id = process % number_of_experiments
    data_id = process // number_of_experiments
    process = experiment_id 

    number_of_rounds = 15

    graph_structure = 'tree10'
    number_of_clients = [5, 1]
    acc_seg = [(40, 100), (100, 100)]
    i = number_of_clients[data_id]
    store_folders = [f'{graph_structure}_{number_of_clients[data_id]}client']
    
    tf = 9000
    train_data = [tf * i] * 8
    int_data = [2 * i, 3 * i, 4 * i, 5 * i, 6 * i, 7 * i, 8 * i, 9 * i]
    num_epochs = [10, 10, 10, 10, 10, 10, 10, 10]
    logger.info(f'Starting the experiment sequence for process {process}: {experiment_id}, {data_id}\n')

    # Change to CausalLearningFederated sub-module
    if os.path.basename(os.getcwd()) != 'CausalLearningFederated':
        os.chdir(os.pardir)

    Experiments.experiment_dsdi_federated(experiment_id=experiment_id,
                                          number_of_rounds=number_of_rounds,
                                          number_of_clients=number_of_clients[data_id],
                                          accessible_segment=acc_seg[data_id],
                                          graph_structure=graph_structure,
                                          store_folder=store_folders[0],
                                          train_functional=train_data[process],
                                          epi_size=int_data[process],
                                          num_epochs=num_epochs[process])

    logger.info(f'Ending the experiment sequence for process {process}\n')


if __name__ == '__main__':
    parallel_experiment()

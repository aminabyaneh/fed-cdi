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
    num_vars = 30
    access_list = [(100, 100)]
    batch_sizes = [50, 60, 70, 80, 90, 100, 110, 120]

    # Federated
    num_rounds = 5
    num_clients = 5
    obs_data_size = 500000
    int_data_size = num_vars * 320 * num_vars
    num_epochs = 3
    folder_name = f'Graph-{graph_type}-{num_vars}'

    Experiments.enco_federated(num_rounds, num_clients, experiment_id, folder_name,
                               access_list[0], obs_data_size,
                               int_data_size, num_epochs, num_vars, graph_type,
                               batch_sizes[experiment_id])

    logger.info(f'Ending the experiment sequence for process {process}\n')


if __name__ == '__main__':
    parallel_experiments_enco()

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

from experiments import Experiments
from logging_settings import logger


def parallel_experiments_enco_int():
    """ A method to handle parallel MPI cluster experiments.
    """
    process = int(str(sys.argv[1]))
    logger.info(f'Starting the experiment sequence for process {process}\n')

    """ Configurations """

    # Id
    experiment_id = process

    # Graph
    graph_types = ["jungle"] #["full", "chain", "jungle", "collider"]
    num_vars = 50

    vars_list = [var_idx for var_idx in range(num_vars)]

    interventions = [[spl.tolist() for spl in np.array_split(vars_list, 5)],
                     [spl.tolist() for spl in np.array_split(vars_list, 10)],
                     [spl.tolist() for spl in np.split(vars_list, [5, 10, 30, 40])],
                     [spl.tolist() for spl in np.split(vars_list, [5, 10, 15, 20, 25, 30, 35, 40, 45])],
                     [spl.tolist() for spl in np.split(vars_list, [2, 4, 20, 40])],
                     [spl.tolist() for spl in np.split(vars_list, [2, 4, 6, 8, 10, 15, 20, 30, 40])],
                     [[c for c in range(50)]],
                     [[c for c in range(25)]]]
    # Federated
    num_rounds = 10
    num_clients = [5, 10, 5, 10, 5, 10, 1, 1]

    accessible_data = (100, 100)
    obs_data_size = 1000 * num_clients[experiment_id]

    p = 1
    int_data_size = 32 * (p * num_vars) * num_clients[experiment_id]

    num_epochs = 2
    repeat = 5

    for graph_type in graph_types:
        folder_name = f'GraphAsym-{graph_type}-{num_vars}'

        for r in range(repeat):
            Experiments.enco_federated_int(interventions[experiment_id], num_rounds, num_clients[experiment_id], experiment_id, r,
                                           folder_name, accessible_data, obs_data_size,
                                           int_data_size, num_epochs, num_vars, graph_type)

        logger.info(f'Ending the experiment sequence for process {process}\n')



def parallel_experiments_enco_int_rnd():
    """ A method to handle parallel MPI cluster experiments.
    """
    process = int(str(sys.argv[1]))

    """ Configurations """
    # Id
    experiment_id = process
    if experiment_id not in [4, 6]:
        return

    # Graph
    edge_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
    num_vars = 50

    vars_list = [var_idx for var_idx in range(num_vars)]

    interventions = [[spl.tolist() for spl in np.array_split(vars_list, 5)],
                     [spl.tolist() for spl in np.array_split(vars_list, 10)],
                     [spl.tolist() for spl in np.split(vars_list, [5, 10, 30, 40])],
                     [spl.tolist() for spl in np.split(vars_list, [5, 10, 15, 20, 25, 30, 35, 40, 45])],
                     [spl.tolist() for spl in np.split(vars_list, [2, 4, 20, 40])],
                     [spl.tolist() for spl in np.split(vars_list, [2, 4, 6, 8, 10, 15, 20, 30, 40])],
                     [[c for c in range(50)]],
                     [[c for c in range(25)]]]
    # Federated
    num_rounds = 10
    num_clients = [5, 10, 5, 10, 5, 10, 1, 1]

    accessible_data = (100, 100)
    obs_data_size = 5000 * num_clients[experiment_id]

    p = 2
    int_data_size = 32 * (p * num_vars) * num_clients[experiment_id]

    num_epochs = 2
    repeat = 5

    for edge_prob in edge_probs:
        logger.info(f'Starting the experiment sequence for edge {edge_prob}\n')
        folder_name = f'GraphAsym-{edge_prob}-{num_vars}'

        for r in range(repeat):
            Experiments.enco_federated_int(interventions[experiment_id], num_rounds, num_clients[experiment_id], experiment_id, r,
                                           folder_name, accessible_data, obs_data_size,
                                           int_data_size, num_epochs, num_vars, graph_type="random", edge_prob=edge_prob)

    logger.info(f'Ending the experiment sequence for process {process}\n')

if __name__ == '__main__':
    parallel_experiments_enco_int()

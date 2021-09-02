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
from config_handle import Parser, Configs


def parallel_experiment():
    """
    This functions is capable on running a series of experiment as a job array on
    the Tuebingen cluster.

    Note: Use the create_job.py function to utilize this.
    Warning: Running on local systems does not yield any results.

    """
    process = int(str(sys.argv[1]))

    p = Parser('default_configuration')
    config_dict = p.load_json_config()

    experiment_id = process % config_dict[Configs.EXPERIMENTS_PER_SET]
    data_id = process // config_dict[Configs.EXPERIMENTS_PER_SET]

    logger.info(f'Starting the experiment sequence for process {process}: {experiment_id}, {data_id}\n')

    # Change to CausalLearningFederated sub-module
    if os.path.basename(os.getcwd()) != 'CausalLearningFederated':
        os.chdir(os.path.join(os.pardir, 'libs', 'CausalLearningFederated'))

    Experiments.experiment_dsdi_federated(experiment_id=experiment_id,
                                          number_of_rounds=config_dict[Configs.NUMBER_OF_ROUNDS],
                                          number_of_clients=config_dict[Configs.NUMBER_OF_CLIENTS][data_id],
                                          accessible_segment=config_dict[Configs.CLIENTS_ACCESSIBLE_DATA][experiment_id],
                                          graph_structure=config_dict[Configs.GRAPH_STRUCTURE],
                                          store_folder=config_dict[Configs.STORE_FOLDERS][data_id],
                                          train_functional=config_dict[Configs.TRAIN_DATA])

    logger.info(f'Ending the experiment sequence for process {process}\n')


if __name__ == '__main__':
    parallel_experiment()

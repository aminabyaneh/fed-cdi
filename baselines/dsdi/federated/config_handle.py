"""
    File name: config_handle.py
    Author: Amin Abyaneh
    Email: aminabyaneh@gmail.com
    Date created: 21/07/2021
    Python Version: 3.8
    Description: Using json configuration files for automated experiments on cluster.
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

import json
import os
from typing import Dict, List


class Configs:
    """
    Key values for the experiment configurations.
    """

    EXPERIMENTS_PER_SET = "experiments_per_set"
    NUMBER_OF_ROUNDS = "number_of_rounds"
    GRAPH_STRUCTURE = "graph_structure"
    CLIENTS_ACCESSIBLE_DATA = "clients_accessible_data"
    NUMBER_OF_CLIENTS = "number_of_clients"
    STORE_FOLDERS = "store_folders"
    TRAIN_DATA = "train_data"


class ComplexEncoder(json.JSONEncoder):
    """A class to encode hierarchical complex objects to JSON format.
    """

    def default(self, obj):
        """The converter method.
        If object is still complex, the method runs toJSON method.
        Args:
            obj (object): The object which should be converted.
        Returns:
            str: The JSON string result of the conversion.
        """
        if hasattr(obj, 'to_json'):
            return obj.to_json()
        else:
            return json.JSONEncoder.default(self, obj)


class Parser:
    """

    A class to encode/decode objects into/from a JSON file.
    """

    def __init__(self, config_name: str = 'default_configuration'):
        """
        Initialize Parser object using folder name.

        Args:
            config_name (str, optional): Folder name indicates the folder name to categorize
            the JSON files. Defaults to 'example'.
        """

        self._config_name: str = config_name
        self.config_dict = dict()

    def store_as_json(self, config_dictionary: Dict):
        """
        Build a json string for an object.

        Args:
            config_dictionary (Dict): A dictionary object to be converted into JSON format.
        """

        json_string = json.dumps(config_dictionary, cls=ComplexEncoder,
                                 sort_keys=False, indent=4,
                                 separators=(',', ': '))

        if os.path.basename(os.getcwd()) == 'federated' or os.path.basename(os.getcwd()) == 'causal':
            os.chdir(os.pardir)

        path = os.path.join('configs', self._config_name + '.json')
        with open(path, "w+") as json_file:
            json_file.write(json_string)

        self.config_dict = config_dictionary

    def load_json_config(self) -> Dict:
        """
        Load a stored config file with the name given in the class initiation.
        """

        if os.path.basename(os.getcwd()) == 'federated' or os.path.basename(os.getcwd()) == 'causal':
            os.chdir(os.pardir)

        # load json file
        path = os.path.join('configs', self._config_name + '.json')
        with open(path, 'r') as f:
            self.config_dict = json.load(f)

        return self.config_dict

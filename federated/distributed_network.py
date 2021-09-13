"""
    File name: distributed_network.py
    Author: Amin Abyaneh
    Email: aminabyaneh@gmail.com
    Date created: 15/04/2021
    Python Version: 3.8

    Description: Designing the components of a distributed network
    based on Pytorch tools.
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
from typing import List, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from logging_settings import logger

""" Commands related to local port resetting:

      1. # fuser 29500/tcp: View all the processes attached to the port.
      2. # kill -9 $(fuser 29500/tcp 2>/dev/null): To kill all of them!
"""

# The array containing all the processes created by multiprocessing module.
process_array: List[torch.multiprocessing.Process] = list()


class Network:
    """
    The network class provides an elaborate interface to handle the topology and
    communications of a federated setup.

    Note: The use of word 'agent' here refers to both clients and server.

    For further information related to built-in Pytorch functions, please refer
    to: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
    """

    def __init__(self, network_size: int, start_method: str = "spawn",
                 backend: dist.Backend = dist.Backend.GLOO):
        """
        Initialize a network class.

        Args:
            network_size (int): Number of agents involved in the network, including
            the server and the clients.

            start_method (int, optional): Change the start method from spawn if
            necessary, otherwise this should work just fine.

            backend (int, optional): The backend of distributed network environment,
            depending on build-time configurations, valid values include mpi, gloo,
            and nccl. Defaults to dist.Backend.GLOO.
        """
        if not dist.is_available():
            raise ModuleNotFoundError("Pytorch distributed package is missing.")

        # Determine the network size
        self.network_size = network_size

        # Backend structure (don't change unless necessary)
        self.backend = backend

        # Create a list of processes
        self.process_array: List = []

        # Set a default server rank
        self.server_rank = 0

        # Initialize the multi-processing
        mp.set_start_method(start_method)

    def init_process(self, agent_rank: int, run_function: Callable):
        """
        Initialize a single agent's process.

        Args:
            agent_rank (int): The rank of the agent to be initialized.
            run_function (Callable): The process associated with this
            agent.
        """

        # The ip/port set for starting the distributed process
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        # Initialize the process group by Pytorch function
        dist.init_process_group(self.backend, rank=agent_rank, world_size=self.network_size)

        # Run the process associated with this agent
        run_function(agent_rank)

    def execute_all_process(self, run_function: Callable):
        """
        Start the execution of each agent's process, simultaneously.

        This function utilizes the Pytorch multiprocessing library to
        develop a concurrent paradigm for all the agents.
        """

        # Detach a thread for each agent
        for rank in range(self.network_size):
            p = mp.Process(target=self.init_process, args=(rank, run_function))
            p.start()
            process_array.append(p)

        # Join the detached threads
        for p in process_array:
            p.join()

    def run_send_recv(self, agent_rank: int):
        """
        Simple send and receive script. The agent with the server rank will initiate
        a message to all other clients through Pytorch send/receive interface.

        Args:
            agent_rank (int): Rank of the current agent.
        """
        tensor = torch.zeros(1)

        # The main agent, i.e., server.
        if agent_rank == self.server_rank:
            tensor += 1

            # Send the tensor to all processes
            for p_rank in range(self.network_size):
                if p_rank is not self.server_rank:
                    dist.send(tensor=tensor, dst=p_rank)

        # The rest of the agents, i.e., clients.
        else:
            # Receive tensor from process 0
            dist.recv(tensor=tensor, src=0)
            logger.info(f'Node {agent_rank} has received {tensor}.')

    def run_broadcast(self, agent_rank: int):
        """
        Demonstrating a distributed program for broadcasting information.

        Args:
            agent_rank (int): Rank of current agent.
        """
        tensor = torch.zeros(1)

        # The main agent, i.e., server.
        if agent_rank == self.server_rank:
            tensor += 1

            logger.info(f'Server {agent_rank} sent broadcast {tensor}')
            dist.broadcast(tensor=tensor, src=self.server_rank)

        # The rest of the agents, i.e., clients.
        else:
            logger.info(f'Agent {agent_rank} tensor value is {tensor}')
            dist.broadcast(tensor=tensor, src=self.server_rank)
            logger.info(f'Agent {agent_rank} received broadcast {tensor}')

    def run_distributed_inference_biagent(self, agent_rank: int, algorithm: str = 'lingam'):
        """
        The agent running this function is instructed to perform a lingam on a specified dataset.

        Args:
            agent_rank (int): The id or rank of the current agent running this distributed
            function.

            algorithm (str): The causal inference algorithm. Defaults to 'lingam'.

        """

        # Phase 0: Preparations

        # Set up the local learning algorithm
        if algorithm == 'lingam':
            causal_model = None
        else:
            raise NotImplementedError

        # Load the associated dataset
        causal_model.load_random_variables(agent_id=agent_rank)

        logger.info(f'The inferred adjacency matrix for node {agent_rank} '
                    f'is: \n {causal_model.original_adjacency_mat}')

        # Phase 1: Learn a local causal model
        causal_model.infer_causal_structure()

        logger.info(f'The inferred adjacency matrix for node {agent_rank} '
                    f'is: \n {causal_model.inferred_adjacency_mat}')

        # Phase 2: Negotiation
        # Phase 3: Post-processing
        # Phase 4: Directing

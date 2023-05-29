"""
    File name: realworld_baselines.py
    Author: Amin Abyaneh
    Email: aminabyaneh@gmail.com
    Date created: 15/08/2022
    Python Version: 3.8
    Description: This is a test for visualization and baselining of real-world datasets: sachs and diabetes.
        The code is built upon ENCO src file: graph_real_world. To utilize a real-world graph for experiments,
        one should pass the graph type object as the name of the dataset to the federated simulation graph, as
        described later in this file.
"""


import sys
import numpy as np
from typing import Dict, List

sys.path.append("../")

from glob import glob
from causal_graphs.graph_visualization import visualize_graph
from causal_graphs.graph_real_world import load_graph_file
from causal_graphs.graph_definition import CausalDAGDataset, CausalDAG

from federated.logging_settings import logger
from federated.causal_learning import ENCOAlg
from federated.federated_simulation import FederatedSimulator


if __name__ == '__main__':

    datasets: Dict[str, CausalDAGDataset] = dict()
    files = sorted(glob("../data/*.bif"))

    for f in files:
        graph = load_graph_file(f)
        print(f, "-> %i nodes, %i categories overall" %
              (graph.num_vars, sum([v.prob_dist.num_categs for v in graph.variables])))

        original_adjacency_mat = graph.adj_matrix
        n_vars = original_adjacency_mat.shape[0]
        logger.debug(f'Global dataset adjacency matrix: \n {original_adjacency_mat.astype(int)}')

        data_obs = graph.sample(batch_size=n_vars * 1000, as_array=True)
        logger.info(f'Shape of global observational data: {data_obs.shape}')

        data_int = ENCOAlg.sample_int_data(graph, n_vars * 50)
        logger.info(f'Shape of global interventional data: {data_int.shape}\n')

        dataset = CausalDAGDataset(original_adjacency_mat, data_obs, data_int)

        print(f'Shape of Obs data: {dataset.data_obs.shape}')
        print(f'Shape of Int data: {dataset.data_int.shape}')
        print(f'Excluded interventions: {graph.exclude_inters}')

        interventions_dict = {0: [var_idx for var_idx in range(n_vars)], 1: [var_idx for var_idx in range(n_vars)]}

        federated_model = FederatedSimulator(interventions_dict, num_clients=2, num_rounds=5, client_parallelism=False)
        federated_model.initialize_clients_data(external_global_dataset=dataset)
        federated_model.execute_simulation(aggregation_method="naive", initial_mass=np.array([16, 16]), alpha=0.2,
            beta=0.3, min_mass=0.1)
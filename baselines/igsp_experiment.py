import os
import sys
import pickle
import argparse

import numpy as np

sys.path.append("../federated")
sys.path.append("../causal_graphs")

from causal_learning import ENCOAlg
from graph_definition import CausalDAGDataset
from logging_settings import logger
from utils import calculate_metrics

from causaldag import igsp, unknown_target_igsp, partial_correlation_suffstat, partial_correlation_test
from graphical_models.rand import directed_erdos, rand_weights
from conditional_independence.ci_tests import MemoizedCI_Tester, hsic_test, kci_test, hsic, kci
from conditional_independence import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester
from conditional_independence import hsic_invariance_test, kci_invariance_test


def prepare_igsp(causal_dag_dataset, alpha=1e-3, alpha_inv=1e-3, ci_test="hsic"):
    """
        Convert ENCO dataset to IGSP output.
    """

    obs_samples = causal_dag_dataset.data_obs
    int_samples = causal_dag_dataset.data_int

    iv_samples_list = list()
    targets_list = list()

    for var_idx in range(causal_dag_dataset.adj_matrix.shape[0]):
        for int_sample in int_samples[var_idx]:
            iv_samples_list.append(int_samples[var_idx])
            targets_list.append([var_idx])

    logger.info(f'Shape of obs_samples {obs_samples.shape}')
    logger.info(f'Shape of iv_samples_list ({len(iv_samples_list)}, {iv_samples_list[0].shape})')

    contexts = {i:s for i,s in enumerate(iv_samples_list)}
    invariance_suffstat = {"obs_samples":obs_samples}
    invariance_suffstat.update(contexts)

    ci_tester = MemoizedCI_Tester(hsic_test, obs_samples, alpha=alpha)
    invariance_tester = MemoizedInvarianceTester(hsic_invariance_test, invariance_suffstat, alpha=alpha_inv)

    setting_list = [dict(interventions=targets) for targets in targets_list]
    nodes = set(range(causal_dag_dataset.adj_matrix.shape[0]))
    return ci_tester, invariance_tester, nodes, setting_list


def build_local_dataset(causal_dag_dataset, client_id: int, num_clients: int, num_vars: int, accessible_p: int = 100):
    """ Build the local dataset for an specific client.
    Args:
        causal_dag_dataset (CausalDAGDataset): The global dataset to be splitted among clients.
        num_clients (int): The total number of clients in the setup.
        num_vars (int): The number of underlying graph nodes.
    """
    adjacency_mat = causal_dag_dataset.adj_matrix
    global_obs_data = causal_dag_dataset.obs_data
    global_int_data = causal_dag_dataset.int_data

    data_length = (global_obs_data.shape[0] // num_clients)
    start_index = data_length * (client_id)

    data_length_acc = int(data_length * (accessible_p / 100))
    end_index = start_index + data_length_acc

    local_obs_data = global_obs_data[start_index: end_index]
    logger.info(f'Client {client_id}: Shape of the local observational data: {local_obs_data.shape}')

    local_int_data: np.ndarray = None
    for var_idx in range(num_vars):
        data_length = (global_int_data.shape[1] // num_clients)
        start_index = data_length * (client_id)

        data_length_acc = int(data_length * (accessible_p / 100))
        end_index = start_index + data_length_acc

        int_sample = global_int_data[var_idx][start_index: end_index]

        local_int_data = np.array([int_sample]) if local_int_data is None \
                                                else np.append(local_int_data,
                                                                np.array([int_sample]),
                                                                axis=0)
    logger.info(f'Client {client_id}: Shape of the local interventional data: {local_int_data.shape}')

    return CausalDAGDataset(adjacency_mat,
                            local_obs_data,
                            local_int_data,
                            exclude_inters=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--obs-samples-size', type=int, default=3000, help='Observational dataset sample size.')
    parser.add_argument('--int-samples-size', type=int, default=20,
                        help='Size of interventional data for all variables, i.e. num_vars * data_samples_per_variable.')
    parser.add_argument('--num-vars', type=int, default=20, help='Number of variables for generating a synthetic dataset.')
    parser.add_argument('--graph-type', type=str, default='full', help='Graph type for data generation process.')
    parser.add_argument('--edge-prob', type=float, default=0.0, help='Edge probability for random graphs.')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed.')
    parser.add_argument('--num-clients', type=int, default=2, help='Number of clients in federated setup.')
    parser.add_argument('--num-rounds', type=int, default=1, help='Number of rounds in federated.')

    opt = parser.parse_args()

    np.random.seed(opt.seed)

    global_causal_dag_dataset = ENCOAlg.build_global_dataset(obs_data_size=opt.obs_samples_size,
                                                             int_data_size=opt.int_samples_size,
                                                             num_vars=opt.num_vars, graph_type=opt.graph_type,
                                                             seed=opt.seed, num_categs=10, edge_prob=opt.edge_prob)

    """ Simple federated 1-round """
    clients_adjacency_matrix = list()

    for client_id in range(opt.num_clients):
        local_causal_dag_dataset = build_local_dataset(global_causal_dag_dataset, client_id, opt.num_clients, opt.num_vars)

        ci_tester, invariance_tester, nodes, setting_list = prepare_igsp(local_causal_dag_dataset)
        est_dag = igsp(setting_list, nodes, ci_tester, invariance_tester, nruns=5)
        dag = est_dag.to_amat()[0]
        logger.info(f'Final DAG by client {client_id}: \n{dag}')

        clients_adjacency_matrix.append(dag)

    """ Calculate the naive aggregation """
    clients_adjacency_matrix = np.asarray(clients_adjacency_matrix)
    aggregated_adj_mat = ((np.sum(clients_adjacency_matrix, axis=1) / opt.num_clients) >= 0.5).astype(int)

    """ Store the results """
    final_results = {
        "clients_adj_matrices": clients_adjacency_matrix,
        "aggregated_adj_mat": aggregated_adj_mat,
        "num_clients": opt.num_clients,
        "graph_type": opt.graph_type,
        "num_vars": opt.num_vars,
        "edge_prob": opt.edge_prob,
        "dataset_sizes": [opt.obs_samples_size, opt.int_samples_size]
    }

    """ Save the list of dags """
    with open(os.path.join('results', 'igsp.pickle')) as f:
        pickle.dump(clients_adjacency_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f'Saving results was successful.')
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





"""GIES algorithm.

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""
import os
import uuid
import warnings
import networkx as nx
from shutil import rmtree
from cdt.causality.graph.model import GraphModel
from pandas import DataFrame, read_csv
from cdt.utils.Settings import SETTINGS
from cdt.utils.R import RPackages, launch_R_script
import numpy as np
import torch


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class GIES(GraphModel):
    """GIES algorithm **[R model]**.

    **Description:** Greedy Interventional Equivalence Search algorithm.
    A score-based Bayesian algorithm that searches heuristically the graph which minimizes
    a likelihood score on the data. The main difference with GES is that it
    accepts interventional data for its inference.

    **Required R packages**: pcalg

    **Data Type:** Continuous (``score='obs'``) or Categorical (``score='int'``)

    **Assumptions:** The output is a Partially Directed Acyclic Graph (PDAG)
    (A markov equivalence class). The available scores assume linearity of
    mechanisms and gaussianity of the data.

    Args:
        score (str): Sets the score used by GIES.
        verbose (bool): Defaults to ``cdt.SETTINGS.verbose``.

    Available scores:
        + int: GaussL0penIntScore
        + obs: GaussL0penObsScore

    .. note::
       Ref:
       D.M. Chickering (2002).  Optimal structure identification with greedy search.
       Journal of Machine Learning Research 3 , 507–554

       A. Hauser and P. Bühlmann (2012). Characterization and greedy learning of
       interventional Markov equivalence classes of directed acyclic graphs.
       Journal of Machine Learning Research 13, 2409–2464.

       P. Nandy, A. Hauser and M. Maathuis (2015). Understanding consistency in
       hybrid causal structure learning.
       arXiv preprint 1507.02608

       P. Spirtes, C.N. Glymour, and R. Scheines (2000).
       Causation, Prediction, and Search, MIT Press, Cambridge (MA)

    Example:
        >>> import networkx as nx
        >>> from cdt.causality.graph import GIES
        >>> from cdt.data import load_dataset
        >>> data, graph = load_dataset("sachs")
        >>> obj = GIES()
        >>> #The predict() method works without a graph, or with a
        >>> #directed or undirected graph provided as an input
        >>> output = obj.predict(data)    #No graph provided as an argument
        >>>
        >>> output = obj.predict(data, nx.Graph(graph))  #With an undirected graph
        >>>
        >>> output = obj.predict(data, graph)  #With a directed graph
        >>>
        >>> #To view the graph created, run the below commands:
        >>> nx.draw_networkx(output, font_size=8)
        >>> plt.show()
    """

    def __init__(self, score='obs', verbose=False):
        """Init the model and its available arguments."""
        if not RPackages.pcalg:
            raise ImportError("R Package pcalg is not available.")

        super(GIES, self).__init__()
        self.scores = {'int': 'GaussL0penIntScore',
                       'obs': 'GaussL0penObsScore'}
        self.arguments = {'{FOLDER}': '/tmp/cdt_gies/',
                          '{FILE}': 'data.csv',
                          '{SKELETON}': 'FALSE',
                          '{GAPS}': 'fixedgaps.csv',
                          '{TARGETS}': 'targets.csv',
                          '{SCORE}': 'GaussL0penIntScore',
                          '{VERBOSE}': 'FALSE',
                          '{LAMBDA}': '1',
                          '{OUTPUT}': 'result.csv'}
        self.verbose = SETTINGS.get_default(verbose=verbose)
        self.score = score

    def run_gies(self, data, fixedGaps=None, targets=None, lambda_gies=None, verbose=True):
        """Setting up and running GIES with all arguments."""
        id = str(uuid.uuid4())
        os.makedirs('/tmp/cdt_gies' + id + '/')
        self.arguments['{FOLDER}'] = '/tmp/cdt_gies' + id + '/'
        self.arguments['{LAMBDA}'] = str(lambda_gies)

        def retrieve_result():
            return read_csv('/tmp/cdt_gies' + id + '/result.csv', delimiter=',').values

        try:
            data.to_csv('/tmp/cdt_gies' + id + '/data.csv', header=False, index=False)
            self.arguments['{SKELETON}'] = 'FALSE'

            if targets is not None:
                targets.to_csv('/tmp/cdt_gies' + id + '/targets.csv', index=False, header=False)
                self.arguments['{INTERVENTION}'] = 'TRUE'
            else:
                self.arguments['{INTERVENTION}'] = 'FALSE'


            gies_result = launch_R_script("{}/gies.R".format(os.path.dirname(os.path.realpath(__file__))),
                                          self.arguments, output_function=retrieve_result, verbose=verbose)

        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_gies' + id + '')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_gies' + id + '/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_gies' + id + '')
        return gies_result

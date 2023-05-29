"""
    File name: baseline_experiments.py
    Author: Amin Abyaneh
    Email: aminabyaneh@gmail.com
    Date created: 23/11/2021
    Python Version: 3.8
    Description: Using IGSP and GIES as baselines.

    TODO: Refactor the file.
"""

import os
import sys
import pickle
import argparse
import os
import uuid

import numpy as np
import pandas as pd

from shutil import rmtree
from cdt.utils.R import launch_R_script

from abc import ABC, abstractmethod

sys.path.append("../federated")
sys.path.append("../causal_graphs")


from causal_learning import ENCOAlg
from graph_definition import CausalDAGDataset
from logging_settings import logger
from utils import calculate_metrics

from causaldag import igsp
from conditional_independence.ci_tests import MemoizedCI_Tester, hsic_test
from conditional_independence import MemoizedInvarianceTester
from conditional_independence import hsic_invariance_test, kci_invariance_test


class BaselineExperiments:
    """Abs
    """
    def __init__(self, obs_samples_size: int, int_samples_size: int,
                 num_vars: int, graph_type: str, edge_prob: float,
                 seed: int, num_clients: int, num_rounds: int):
        """ Initialize a baseline experiment.

        Args:
            obs_samples_size (int): Number of samples for observational data. Defaults to 5000.
            int_samples_size (int): Number of samples for interventional data. Defaults to 40.
            num_vars (int): Number of variables of the generator graph. Defaults to 20.
            graph_type (str): Type of the data generator graph. Defaults to "full".
            edge_prob (float): Edge probability in case the graph type is random. Defaults to 0.0.
            seed (int): Random seed of the experiment. Defaults to 0.
            num_clients (int): Total number of clients. Defaults to 2.
            num_rounds (int): Total number of federated rounds. Defaults to 1.
        """
        self.obs_samples_size = obs_samples_size
        self.int_samples_size = int_samples_size
        self.num_vars = num_vars
        self.graph_type = graph_type
        self.edge_prob = edge_prob
        self.seed = seed
        self.num_clients = num_clients
        self.num_rounds = num_rounds

        np.random.seed(self.seed)

        self.global_dataset = ENCOAlg.build_global_dataset(obs_data_size=self.obs_samples_size,
                                                           int_data_size=self.int_samples_size,
                                                           num_vars=self.num_vars,
                                                           graph_type=self.graph_type,
                                                           seed=self.seed, num_categs=10,
                                                           edge_prob=self.edge_prob)

    @abstractmethod
    def run(self):
        """Running a baseline experiment.

        Raises:
            NotImplementedError: Raised to warn about not implementing a required method.
        """
        raise NotImplementedError

    @abstractmethod
    def build_local_dataset(self):
        """Build dataset for a baseline experiment.

        Raises:
            NotImplementedError: Raised to warn about not implementing a required method.
        """
        raise NotImplementedError


class GIES(BaselineExperiments):
    """Class to assess GIES performance in a federated setup.
    """
    def __init__(self, obs_samples_size: int = 5000, int_samples_size: int = 40,
                 num_vars: int = 5, graph_type: str = "full", edge_prob: float = 0.0,
                 seed: int = 0, num_clients: int = 2, num_rounds: int = 1):
        """Initialization of experiments for GIES method.
        TODO: Check if Nan is the right way to go for observational samples target list.

        Args:
            obs_samples_size (int, optional): Number of samples for observational data. Defaults to 5000.
            int_samples_size (int, optional): Number of samples for interventional data. Defaults to 40.
            num_vars (int, optional): Number of variables of the generator graph. Defaults to 20.
            graph_type (str, optional): Type of the data generator graph. Defaults to "full".
            edge_prob (float, optional): Edge probability in case the graph type is random. Defaults to 0.0.
            seed (int, optional): Random seed of the experiment. Defaults to 0.
            num_clients (int, optional): Total number of clients. Defaults to 2.
            num_rounds (int, optional): Total number of federated rounds. Defaults to 1.
        """
        super().__init__(obs_samples_size, int_samples_size, num_vars, graph_type,
                         edge_prob, seed, num_clients, num_rounds)
        logger.info('GIES baseline class initialized.')

    @abstractmethod
    def run(self):
        """Run a single GIES experiment and store the results.
        """

        """ Simple federated 1-round """
        clients_adjacency_matrix = list()

        for client_id in range(self.num_clients):
            data, targets = self.build_local_dataset(client_id)
            dag = self.run_gies(data, targets, del_tmp=True)

            logger.info(f'Final DAG by client {client_id}: \n{dag}')
            clients_adjacency_matrix.append(dag)

        """ Calculate the naive aggregation """
        clients_adjacency_matrix = np.asarray(clients_adjacency_matrix)
        aggregated_adj_mat = ((np.sum(clients_adjacency_matrix, axis=1) / self.num_clients) >= 0.5).astype(int)

        """ Store the results """
        final_results = {
            "ground_truth_adj_mat": self.global_dataset.adj_matrix,
            "clients_adj_mats": clients_adjacency_matrix,
            "aggregated_adj_mat": aggregated_adj_mat,
            "num_clients": self.num_clients,
            "graph_type": self.graph_type,
            "num_vars": self.num_vars,
            "edge_prob": self.edge_prob,
            "dataset_sizes": [self.obs_samples_size, self.int_samples_size],
            "final_metrics": calculate_metrics(aggregated_adj_mat, self.global_dataset.adj_matrix)
        }

        """ Save the list of dags """
        results_file_name = f'gies-{self.num_clients}-{self.num_vars}-{self.graph_type}-{self.edge_prob}-{self.seed}.pickle'

        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.makedirs('results', exist_ok=True)

        with open(os.path.join('results', results_file_name), 'wb') as f:
            pickle.dump(final_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f'Saving results was successful.')

    def run_gies(self, data, targets, lambda_gies=1, verbose=True, del_tmp: bool = False):
        """ Setting up and running GIES with interventional and observational data.
        """

        id = str(uuid.uuid4())
        data_path = os.path.join('tmp', f'cdt_gies{id}')

        arguments = {'{FOLDER}': data_path,
                    '{FILE}': '/data.csv',
                    '{SKELETON}': 'FALSE',
                    '{GAPS}': 'fixedgaps.csv',
                    '{TARGETS}': '/targets.csv',
                    '{SCORE}': 'GaussL0penIntScore',
                    '{VERBOSE}': 'FALSE',
                    '{LAMBDA}': '1',
                    '{OUTPUT}': '/result.csv'}

        os.makedirs(data_path)
        arguments['{FOLDER}'] = data_path
        arguments['{LAMBDA}'] = str(lambda_gies)

        def retrieve_result():
            return pd.read_csv(os.path.join(data_path, 'result.csv'), delimiter=',').values

        data.to_csv(os.path.join(data_path, 'data.csv'), header=False, index=False)
        arguments['{SKELETON}'] = 'FALSE'

        if targets is not None:
            if targets.shape[1] == 1:
                targets['dummy'] = np.nan
            targets.to_csv(os.path.join(data_path, 'targets.csv'), index=False, header=False)
            arguments['{INTERVENTION}'] = 'TRUE'

        logger.info('Launching the R script.')
        gies_result = launch_R_script("{}/dcdi/gies/gies.R".format(os.path.dirname(os.path.realpath(__file__))),
                                        arguments, output_function=retrieve_result, verbose=verbose)

        if del_tmp:
            rmtree(data_path)
            logger.info('Temporary files are now deleted.')

        return gies_result

    @abstractmethod
    def build_local_dataset(self, client_id: int, accessible_p: int = 100):
        """ Build the local dataset for an specific client.

        Args:
            client_id (int): The client to whom the generated local dataset belongs.
            accessible_p (int, optional): Accessibe percentage of the client's data fragment.
            Defaults to 100.

        Returns:
            pd.DataFrame, pd.DataFrame: data, targets
        """
        adjacency_mat = self.global_dataset.adj_matrix
        global_obs_data = self.global_dataset.data_obs
        global_int_data = self.global_dataset.data_int

        data_length = (global_obs_data.shape[0] // self.num_clients)
        start_index = data_length * (client_id)

        data_length_acc = int(data_length * (accessible_p / 100))
        end_index = start_index + data_length_acc

        local_obs_data = global_obs_data[start_index: end_index]
        logger.info(f'Client {client_id}: Shape of the local observational data: {local_obs_data.shape}.')

        local_int_data: np.ndarray = None
        for var_idx in range(self.num_vars):
            data_length = (global_int_data.shape[1] // self.num_clients)
            start_index = data_length * (client_id)

            data_length_acc = int(data_length * (accessible_p / 100))
            end_index = start_index + data_length_acc

            int_sample = global_int_data[var_idx][start_index: end_index]

            local_int_data = np.array([int_sample]) if local_int_data is None \
                                                    else np.append(local_int_data,
                                                                    np.array([int_sample]),
                                                                    axis=0)
        logger.info(f'Client {client_id}: Shape of the local interventional data: {local_int_data.shape}.')

        n_variables = local_int_data.shape[0]
        n_samples_per_variable = local_int_data.shape[1]

        obs_data_size = local_obs_data.shape[0]
        int_data_size = n_variables * n_samples_per_variable

        data_np = local_obs_data
        targets_np = np.zeros(obs_data_size + int_data_size)

        no_intervention_indices = np.arange(0, local_obs_data.shape[0])
        targets_np[no_intervention_indices] = np.nan

        for var_idx in range(n_variables):
            data_np = np.append(data_np, local_int_data[var_idx], axis=0)
            var_idx_intervention_indices = np.arange(var_idx * n_samples_per_variable, (var_idx + 1) * n_samples_per_variable)
            targets_np[var_idx_intervention_indices + obs_data_size] = var_idx

        data = pd.DataFrame(data_np)
        targets = pd.DataFrame(targets_np)

        logger.info(f'Data shape is {data.shape}.')
        logger.info(f'Targets shape is {targets.shape}.')

        return data, targets


class IGSP(BaselineExperiments):
    """Class to assess IGSP performance in a federated setup.
    """
    def __init__(self, obs_samples_size: int = 5000, int_samples_size: int = 40,
                 num_vars: int = 20, graph_type: str = "full", edge_prob: float = 0.0,
                 seed: int = 0, num_clients: int = 2, num_rounds: int = 1):
        """Initialization of experiments for IGSP method.

        Args:
            obs_samples_size (int, optional): Number of samples for observational data. Defaults to 5000.
            int_samples_size (int, optional): Number of samples for interventional data. Defaults to 40.
            num_vars (int, optional): Number of variables of the generator graph. Defaults to 20.
            graph_type (str, optional): Type of the data generator graph. Defaults to "full".
            edge_prob (float, optional): Edge probability in case the graph type is random. Defaults to 0.0.
            seed (int, optional): Random seed of the experiment. Defaults to 0.
            num_clients (int, optional): Total number of clients. Defaults to 2.
            num_rounds (int, optional): Total number of federated rounds. Defaults to 1.
        """
        super().__init__(obs_samples_size, int_samples_size, num_vars, graph_type,
                         edge_prob, seed, num_clients, num_rounds)
        logger.info('IGSP baseline class initialized.')

    @abstractmethod
    def run(self):
        """Run a single IGSP experiment and store the results.
        """

        """ Simple federated 1-round """
        clients_adjacency_matrix = list()

        for client_id in range(self.num_clients):
            local_causal_dag_dataset = self.build_local_dataset(client_id)

            ci_tester, invariance_tester, nodes, setting_list = self.prepare_igsp(local_causal_dag_dataset)
            est_dag = igsp(setting_list, nodes, ci_tester, invariance_tester, nruns=5)
            dag = est_dag.to_amat()[0]
            logger.info(f'Final DAG by client {client_id}: \n{dag}')

            clients_adjacency_matrix.append(dag)

        """ Calculate the naive aggregation """
        clients_adjacency_matrix = np.asarray(clients_adjacency_matrix)
        aggregated_adj_mat = ((np.sum(clients_adjacency_matrix, axis=1) / self.num_clients) >= 0.5).astype(int)

        """ Store the results """
        final_results = {
            "ground_truth_adj_mat": self.global_dataset.adj_matrix,
            "clients_adj_mats": clients_adjacency_matrix,
            "aggregated_adj_mat": aggregated_adj_mat,
            "num_clients": self.num_clients,
            "graph_type": self.graph_type,
            "num_vars": self.num_vars,
            "edge_prob": self.edge_prob,
            "dataset_sizes": [self.obs_samples_size, self.int_samples_size],
            "final_metrics": calculate_metrics(aggregated_adj_mat, self.global_dataset.adj_matrix)
        }

        """ Save the list of dags """
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        os.makedirs('results', exist_ok=True)

        results_file_name = f'igsp-{self.num_clients}-{self.num_vars}-{self.graph_type}-{self.edge_prob}.pickle'
        with open(os.path.join('results', results_file_name), 'wb') as f:
            pickle.dump(final_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f'Saving results was successful.')

    def prepare_igsp(self, local_causal_dag_dataset, alpha=1e-3, alpha_inv=1e-3, ci_test="hsic"):
        """
            Convert ENCO dataset to IGSP output.
        """

        obs_samples = local_causal_dag_dataset.data_obs
        int_samples = local_causal_dag_dataset.data_int

        iv_samples_list = list()
        targets_list = list()

        for var_idx in range(local_causal_dag_dataset.adj_matrix.shape[0]):
            for int_sample in int_samples[var_idx]:
                # TODO: FIX THIS
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
        nodes = set(range(local_causal_dag_dataset.adj_matrix.shape[0]))
        return ci_tester, invariance_tester, nodes, setting_list

    @abstractmethod
    def build_local_dataset(self, client_id: int, accessible_p: int = 100):
        """ Build the local dataset for an specific client.

        Args:
            client_id (int): The client to whom the generated local dataset belongs.
            accessible_p (int, optional): Accessible part of client's dataset fragment.
            Defaults to 100.

        Returns:
            CausalDAGDateset: A dataset containing interventional and observational data.
        """
        adjacency_mat = self.global_dataset.adj_matrix
        global_obs_data = self.global_dataset.obs_data
        global_int_data = self.global_dataset.int_data

        data_length = (global_obs_data.shape[0] // self.num_clients)
        start_index = data_length * (client_id)

        data_length_acc = int(data_length * (accessible_p / 100))
        end_index = start_index + data_length_acc

        local_obs_data = global_obs_data[start_index: end_index]
        logger.info(f'Client {client_id}: Shape of the local observational data: {local_obs_data.shape}')

        local_int_data: np.ndarray = None
        for var_idx in range(self.num_vars):
            data_length = (global_int_data.shape[1] // self.num_clients)
            start_index = data_length * (client_id)

            data_length_acc = int(data_length * (accessible_p / 100))
            end_index = start_index + data_length_acc

            int_sample = global_int_data[var_idx][start_index: end_index]

            local_int_data = np.array([int_sample]) if local_int_data is None \
                                                    else np.append(local_int_data,
                                                                    np.array([int_sample]),
                                                                    axis=0)

        logger.info(f'Client {client_id}: Shape of the local interventional data: {local_int_data.shape}')
        return CausalDAGDataset(adjacency_mat, local_obs_data, local_int_data, exclude_inters=None)


if __name__ == '__main__':
    logger.info('Running baselines test...')
    gies_module = GIES()
    gies_module.run()


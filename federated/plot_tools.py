"""
    File name: plot_tools.py
    Author: Amin Abyaneh
    Email: aminabyaneh@gmail.com
    Date created: 25/04/2021
    Python Version: 3.8
    Description: Extracting the execution data by plots.
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
from typing import Dict, List

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from logging_settings import logger
from utils import retrieve_dsdi_stored_data, evaluate_inferred_matrix


def set_plot_styles(title: str, x_label: str, y_label: str, save_file_name: str,
                    legend_location: str = 'upper right', legend_labels: str = None,
                    title_size: int = 20, labels_size: int = 15, ticks_size: int = 15,
                    legend_size: int = 10, xticks_rotation: int = 45):
    """
    Set the plot styles ranging from title, labels and legends all the way to

    Args:
        title (str): The plot title.
        x_label (str): Label of the x axis.
        y_label (str): Label of the y axis.

        save_file_name (str): Where and in what name do you like to save the figure.
        legend_location (str): Optimize the legends locations. Defaults to 'upper right'.
        legend_labels (str): If label is available, change this from None to the label itself.
        Defaults to None.

        title_size (int): Size of the figures title. Defaults to 20.
        labels_size (int): Size of the labels. Defaults to 15.
        ticks_size (int): Ticks size. Defaults to 15.
        legend_size (int): Size of legends. Defaults to 10.
        xticks_rotation (int): The rotation of X-axis labels. Defaults to 45.
    """

    if legend_labels is None:
        plt.legend(ncol=1, loc=legend_location, fontsize=legend_size)
    else:
        plt.legend(labels=legend_labels, loc=legend_location, fontsize=legend_size)

    plt.title(title, fontsize=title_size)
    plt.xlabel(x_label, fontsize=labels_size)
    plt.ylabel(y_label, fontsize=labels_size)

    plt.xticks(fontsize=ticks_size, rotation=xticks_rotation)
    plt.yticks(fontsize=ticks_size)

    plt.savefig(save_file_name, bbox_inches='tight')
    plt.show()


def plot_multiple_experiment_sets(experiments_directory: str, baseline_mat: np.ndarray,
                                  experiment_set_id_dict: Dict[str, List[int]],
                                  number_of_rounds: int, labels_dict: Dict[str, List[str]],
                                  colors_dict: Dict[str, List[str]],
                                  fig_name: str = 'compare_performance.png', metric='ED',
                                  title: str = 'Performance'):
    """
    Plot the performance on the server side for more than one set of experiments. Best if you need to compare multiple
    different sets of experiments.

    Note: The matrices are parsed for each client individually, so later experiments on aggregation step could be
    performed easier. The aggregation step happens before plotting the error or performance.

    Args:
        title (str): The title of the plot.
        experiments_directory (str): The directory where all the experiments are stored.
        baseline_mat (str): The ground truth for the structure learning algorithm.
        experiment_set_id_dict (Dict[str, str]): Indicating a list of experiment Ids corresponding to a experiment set
        to be plotted.

        number_of_rounds (int): For how many rounds do you want to plot each experiment. Should be less than the
        total number of rounds that you've experimented on.

        labels_dict (Dict[str, List[str]]): Label for each curve or experiment.
        colors_dict (Dict[str, List[str]]): Color of each curve or experiment.
        fig_name (str): Name of the figure to be saved. Defaults to 'performance.png'. Change .png suffix to store in
        other formats.

        metric (str): The performance metric. Refer to utils.py. Defaults to 'ED' (Euclidean Distance).
    """

    plt.figure(figsize=(12, 12))
    seaborn.set_style("darkgrid")

    rounds = [round_id for round_id in range(number_of_rounds)]

    for experiment_name in experiment_set_id_dict:
        experiment_directory = os.path.join(experiments_directory, experiment_name)

        for index, experiment_id in enumerate(experiment_set_id_dict[experiment_name]):

            metric_values: List[float] = list()

            for round_id in range(number_of_rounds):
                aggregated_adjacency_matrix: np.ndarray = None
                access_sum: int = 0

                for data in retrieve_dsdi_stored_data(experiment_directory, experiment_id, round_id):
                    if aggregated_adjacency_matrix is None:
                        aggregated_adjacency_matrix = data[0] * data[1]
                    else:
                        aggregated_adjacency_matrix += data[0] * data[1]

                    print(f'Acquired in {experiment_id}:{round_id} \n {data[0]}, {data[1]}')
                    access_sum += data[0]

                inferred_matrix: np.ndarray = aggregated_adjacency_matrix / access_sum
                logger.info(f'Inferred matrix at {experiment_id}:{round_id} \n{inferred_matrix}')
                evaluation_dict = evaluate_inferred_matrix(baseline_mat, inferred_matrix)

                metric_values.append(evaluation_dict[metric])

            plt.plot(rounds, metric_values, label=labels_dict[experiment_name][index], color=colors_dict[experiment_name][index])
            plt.scatter(rounds, metric_values, c=colors_dict[experiment_name][index])

    plt.xticks(rounds)
    set_plot_styles(title=title, x_label='Round',
                    y_label='Euclidean Distance' if metric == 'ED' else 'Hamming Distance',
                    save_file_name=fig_name)


if __name__ == '__main__':
    ground_truth_chain3 = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    ground_truth_chain10 = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

    ground_truth_full10 = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])

    ground_truth_collider10 = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])

    ground_truth_collider5 = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0],
                                         [1.0, 1.0, 1.0, 1.0, 0.0]])


    ground_truth_jungle10 = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    graph_type = 'collider10'
    client_number = 1
    client_numbers = [1, 10]
    metric = 'SHD'
    baseline = ground_truth_collider10
    single = True
    number_of_rounds = 15

    # Collider10: 5: [0, 1, 3, 4, 2] {3, 4, 6, 7, 10}, 1: [1, 3, 0, 2] {3, 6, 8, 12}
    if single:
        plot_multiple_experiment_sets(experiments_directory=f'../../FedCL_Data', baseline_mat=baseline,
                                      experiment_set_id_dict={f'{graph_type}_{client_number}client': [1, 3, 0, 2]},
                                      number_of_rounds=number_of_rounds,
                                      labels_dict={f'{graph_type}_{client_number}client': ['3 Int. Batch', '6 Int. Batch', '8 Int. Batch', '12 Int. Batch', '10 Int. Batch']},
                                      colors_dict={f'{graph_type}_{client_number}client': ['blue', 'brown', 'purple', 'green', 'cyan']},
                                      fig_name=f'{graph_type}_{client_number}client_{metric}{number_of_rounds}.png', metric=metric,
                                      title=f'{graph_type} {client_number}-Client')
    else:
        plot_multiple_experiment_sets(experiments_directory=f'../../FedCL_Data', baseline_mat=baseline,
                                      experiment_set_id_dict={f'{graph_type}_{client_numbers[0]}client': [1, 2],
                                                              f'{graph_type}_{client_numbers[1]}client': [1, 2]},
                                      number_of_rounds=number_of_rounds,
                                      labels_dict={
                                          f'{graph_type}_{client_numbers[0]}client':
                                              [f'{client_numbers[0]}-Client 5%', f'{client_numbers[0]}-Client 10%'],
                                          f'{graph_type}_{client_numbers[1]}client':
                                              [f'{client_numbers[1]}-Client 3-5%', f'{client_numbers[1]}-Client 5-10%']},
                                      colors_dict={
                                          f'{graph_type}_{client_numbers[0]}client': ['blue', 'darkblue'],
                                          f'{graph_type}_{client_numbers[1]}client': ['violet', 'purple']},
                                      fig_name=f'{graph_type}_{client_numbers[0]}client_vs_{client_numbers[1]}client_{metric}{number_of_rounds}.png',
                                      metric=metric,
                                      title=f'{graph_type} {client_numbers[0]}-Client vs. {client_numbers[1]}-Client')

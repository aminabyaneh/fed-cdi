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
import pickle
from typing import Dict, List

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from logging_settings import logger


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


def plot_enco_round_results(folder_name: str, experiment_ids: List[int], labels: List[str],
                            colors: List[str], title: str = 'default', fig_name: str = 'default.png',
                            metric: str = "SHD"):

    plt.figure(figsize=(12, 12))
    seaborn.set_style("darkgrid")

    for index, experiment_id in enumerate(experiment_ids):
        dir = os.path.join(os.pardir, 'data', folder_name, f'results_{experiment_id}.pickle')
        with open(dir, 'rb') as handle:
            stored_results_dict = pickle.load(handle)

        final_results = stored_results_dict['priors']

        rounds = [round_id for round_id in range(len(final_results))]
        metrics = [results_dict[metric] for results_dict in final_results]

        logger.info(f'Experiment {experiment_id} metrics: {metrics}')
        plt.plot(rounds, metrics, label=labels[index], color=colors[index])
        plt.scatter(rounds, metrics, c=colors[index])

    plt.xticks(rounds)
    set_plot_styles(title=title, x_label='Round Id',
                    y_label=metric, save_file_name=fig_name)


def plot_enco_clients_results(folder_name: str, experiment_id: int, labels: List[str],
                              colors: List[str], title: str = 'default', fig_name: str = 'default.png',
                              metric: str = "SHD"):

    plt.figure(figsize=(12, 12))
    seaborn.set_style("darkgrid")

    dir = os.path.join(os.pardir, 'data', folder_name, f'results_{experiment_id}.pickle')
    with open(dir, 'rb') as handle:
        stored_results_dict = pickle.load(handle)

    for client_id in stored_results_dict:
        if client_id == 'priors':
            break

        rounds = [round_id for round_id in range(len(stored_results_dict[client_id]))]
        metrics = [results_dict[metric] for results_dict in stored_results_dict[client_id]]

        logger.info(f'Experiment {experiment_id} client {client_id} metrics: {metrics}')
        plt.plot(rounds, metrics, label=labels[client_id], color=colors[client_id])
        plt.scatter(rounds, metrics, c=colors[client_id])

    plt.xticks(rounds)
    set_plot_styles(title=title, x_label='Round Id',
                    y_label=metric, save_file_name=fig_name)

if __name__ == '__main__':
    plot_enco_round_results('tests', [0], ['multiple clients'], ['blue'])
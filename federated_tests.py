# -*- coding: utf-8 -*-
import os
import sys

"""We import PyTorch and Numpy on which the ENCO implementation is based on."""

# Commented out IPython magic to ensure Python compatibility.
import torch
import numpy as np

"""## Causal Graphs

First, we take a look at how we can generate causal graphs and interact with them.
All functionalities for this purpose have been summarized in the folder `causal_graphs`,
and we import the most important functions below.
"""

from causal_graphs.graph_definition import CausalDAGDataset  # Base class of causal graphs
from causal_graphs.graph_generation import generate_categorical_graph, get_graph_func  # Functions for generating new graphs
from causal_graphs.graph_visualization import visualize_graph  # Plotting the graph in matplotlib
from causal_graphs.variable_distributions import _random_categ

from federated.logging_settings import logger

"""Every graph is represented as a `CausalDAG` object that contains a list of variables and
an adjacency matrix/edge list to represent the graph structure. The conditional distributions
are part of the variable objects. To sample a new data point, we iterate through the variables
in the causal order and sample one by one. To demonstrate this, let us first generate an arbitrary
graph. This can be done with the `generate_categorical_graph` function, and we can specify
the desired graph structure with `get_graph_func`:"""


num_vars = 20
graph_type = "full"

graph = generate_categorical_graph(num_vars=num_vars,
                                   min_categs=10,
                                   max_categs=10,
                                   use_nn=True,
                                   graph_func=get_graph_func(graph_type),
                                   seed=0)

"""This function call creates a new graph with 8 variables, each having a distribution over 10
categories, and the graph structure is generated randomly by sampling an edge between any pair
of variables with a probability of 0.4. The seed ensures that the graph generation is
reproducible. To generate other graph structures, simply replace the string `'random'`
by e.g. `'chain'` or `'jungle'`.

To get an intuition of what the graph actually looks like, we can print it:
"""

logger.info(graph)

"""The variables are named alphabetically, and we have 10 edges.
The edges are listed below the first line, e.g., we have an edge from D to E,
and an edge from D to G. Alternatively, we can also plot the graph with matplotlib: """

visualize_graph(graph, figsize=(4, 4), filename="federated_test.png")

"""To sample from a graph, we use the function `CausalDAG.sample`:"""

logger.info(f'Acquired sample from the graph: \n {graph.sample()}')


"""Sampling with interventions is supported by passing a dictionary with the intended
interventions. The interventions can be imperfect, i.e. a new distribution, or perfect,
i.e. constant values. We demonstrate here a perfect intervention on the variable C:"""

logger.info(f'Acquired sample from the graph: \n '
            f'{graph.sample(interventions={"C": np.array([0])})}')

"""Graphs can be saved and loaded with the function `save_to_file`
and `CausalDAG.load_from_file`. To save the graph as a set of
observational and interventional dataset, you can use the
function `export_graph` from `graph_export.py`. We used this
functionality to export the data to apply other causal discovery methods on.
Graphs in the `.bif` format, as from the BnLearn repository, can be loaded via the
function `load_graph_file` in `graph_real_world.py`.


## Offline Datasets
"""
# Size of the global dataset
observational_dataset_size = 30000
interventional_dataset_size = num_vars * 64

# Adjacency matrix
adj_matrix = graph.adj_matrix
logger.info(f'Global dataset adjacency matrix: \n {adj_matrix}')

# Observational data
data_obs = graph.sample(batch_size=observational_dataset_size, as_array=True)
logger.info(f'Shape of observational data: {data_obs.shape}')
logger.info(f'Observational data: {data_obs}')


# Interventional data
data_int: np.ndarray = None

for var_idx in range(len(graph.variables)):
    # Select variable to intervene on
    var = graph.variables[var_idx]

    # Soft, perfect intervention => replace p(X_n) by random categorical
    # Scale is set to 0.0, which represents a uniform distribution.
    int_dist = _random_categ(size=(var.prob_dist.num_categs,), scale=0.0, axis=-1)

    # Sample from interventional distribution
    value = np.random.multinomial(n=1, pvals=int_dist,
                                  size=(interventional_dataset_size // len(graph.variables),))
    value = np.argmax(value, axis=-1)  # One-hot to index

    intervention_dict = {var.name: value}
    logger.info(f'Intervention dictionary: \n {intervention_dict}')

    int_sample = graph.sample(interventions=intervention_dict,
                              batch_size=(interventional_dataset_size // len(graph.variables)),
                              as_array=True)
    logger.info(f'Interventional data sample: \n {int_sample}')


    data_int = np.array([int_sample]) if data_int is None \
                                      else np.append(data_int, np.array([int_sample]), axis=0)
    logger.info(f'Shape of interventional data: {data_int.shape}')


global_dataset_dag = CausalDAGDataset(adj_matrix, data_obs, data_int)


from causal_discovery.enco import ENCO

enco_module = ENCO(graph=global_dataset_dag)
if torch.cuda.is_available():
    logger.info('Found Cuda device!')
    enco_module.to(torch.device('cuda:0'))

predicted_adj_matrix = enco_module.discover_graph(num_epochs=10)

"""
## Causal Discovery with ENCO

The graph objects explained above are used to implement the structure learning with
ENCO in the folder `causal_discovery`. To run ENCO on such a graph, we simply need
to create an `ENCO` object, and run the structure learning via the `discover_graph`
function:
"""

from causal_discovery.enco import ENCO

obs_data_size = [3500]
int_batches_count = [16, 32, 64, 128]

results_dict = dict()

for obs_s in obs_data_size:
    for int_s in int_batches_count:
        enco_module = ENCO(graph=graph, batch_size=int_s, dataset_size=obs_s)
        if torch.cuda.is_available():
            logger.info('Found Cuda device!')
            enco_module.to(torch.device('cuda:0'))

        predicted_adj_matrix = enco_module.discover_graph(num_epochs=2)

        metrics = enco_module.get_metrics()
        results_dict[str(obs_s) + ":" + str(int_s)] = metrics["SHD"]

logger.info(f'Results dictionary: \n {results_dict}')

logger.info(f'*** Debug information *** \n Final Gamma: \n {enco_module.gamma} \n'
            f'Final Theta: \n {enco_module.theta}')

"""After every epoch, the metrics of comparing the current prediction to the ground truth graph are printed out.
In the case of the small graph we created above, ENCO finds the graph quite quickly.
The return value is the predicted adjacency matrix, and can be passed to a new graph object if you want to
visualize the prediction. Hyperparameters for the structure learning process can be passed to the ENCO object
in the init-function.

This completes the quick guide through the code. To run experiments on a larger scale,
we recommend to use the python files provided in the `experiments` folder. Further, the commands
to reproduce the experiments in the paper are provided in `experiments/run_scripts/`.
"""

# Reproduce the results inserting prior for sanity check

import torch.nn as nn
prior_structure_mat = enco_module.gamma

new_enco_module = ENCO(graph=graph, prior_info=prior_structure_mat)

if torch.cuda.is_available():
    logger.info('Found Cuda device!')
    enco_module.to(torch.device('cuda:0'))

predicted_adj_matrix = enco_module.discover_graph(num_epochs=2)

logger.info(f'*** Debug information *** \n Final Gamma:'
            f'\n {enco_module.gamma} \n Final Theta: \n {enco_module.theta}')

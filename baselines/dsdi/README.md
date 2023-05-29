

# Causal Learning - Federated Setup 

The Causal Federated Learning project aims at designing a Cross-Silo Federated Causal Learning algorithms,
capable of extracting the underlying Directed Acyclic Graph (DAG) of a distributed dataset, where each agent
only has limited access to a specific set of dataset variables.


Based on the DSDI repository:  https://github.com/nke001/causal_learning_unknown_interventions.

## Motivation

Federated Learning trains an algorithm across multiple decentralized devices holding local data samples, 
without exchanging the private information. Data is generated locally and remains de-centralised. Each 
client stores its own data and cannot read the data of other clients. Data is not independently or 
identically distributed. 

Moreover, privacy is an indisputably crucial matter to the world of causal inference. Considering the potential
of federated learning to preserve privacy and building up sophisticated models simultaneously, one might wonder whether
it is possible to adopt causal learning algorithms to the federated paradigm, aiming at developing a novel and distributed 
generation of causal inference methods.


## Installation 

1. This code is based on Pytorch. The conda enviroment for running this code can be installed as follows,

```
conda env create -f environment.yml
pip install -e .

```
**Important:** CPU features required: AVX2, FMA (Intel Haswell+) 


---
### Multi-client Simulation

Refer to experiments.py under the federated folder for further details.

--- 
### Single run of the core DSDI method

```
# chain3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 --limit-samples 500 -N 2 -p chain3  

# fork3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 --limit-samples 500 -N 2 -p fork3

# collider3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 --limit-samples 500 -N 2 -p collider3

# confounder3
python run.py train --seed 1 --train_functional 10000 --mopt adam:5e-2,0.9 --gopt adam:5e-3,0.1 -v 500 --lsparse 0.1 --bs 256 --ldag 0.5 --predict 100 --temperature 1 --limit-samples 500 -N 2 -p confounder3
```


`--seed` specifies the random seed

`--mopt` specifies the optimizer and learning rate used to train the functional parameters

`--gopt` specifies the optimizer and learning rate used to train the structural parameters

`--predict` specifies the number of samples used for predicting the intervened node. Specifying 0 for this argument uses the groundtruth intervention node.

`--temperature` specifies the temperature setting fot the softmax for the groundtruth structured causal model.

`-N` specifies the number of categories for the categorical distribution

`-M` specifies the number of discrete random variables

`--graph` allows one to specify via the command-line several causal DAG skeletons.

`-p` specifies, by name, one of several `--graph` presets for groundtruth causal graphs (e.g. `chain3`).

`--train_functional` specifies how many iterations to train the functional parameters.

`--limit-samples` specifies the number of samples used per intervention. Suggest to use 500 for graphs of size < 10 and 1000 for graphs size between 10 and 15.


By default, the models and log files are stored in the `work` directory.
 

### How to incorporate prior knowledge / belief about the graph structure
Add following parameter to the training call
```
--gammaBelief <path_to_belief>.npy
```
**Requirement** <file>.npy should contain a proabilistic belief matrix of the same shape as the underlying adjacency matrix with values in the range [0,1]
---

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests to us.




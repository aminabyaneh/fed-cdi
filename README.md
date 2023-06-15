## FED-CD: Federated Causal Discovery From Interventional and Observational Data

The Causal Federated Learning project aims at designing a Cross-Silo Federated Causal Learning algorithms, capable of extracting the underlying Directed Acyclic Graph (DAG) of a distributed dataset, where each agent only has limited access to a specific set of dataset variables and a subset of intervened variables.

For further elaboration and details, you can refer to our [paper](https://arxiv.org/abs/2211.03846).

## Motivation

Federated Learning trains an algorithm across multiple decentralized devices holding local data samples, without exchanging the private information. Data is generated locally and remains decentralized. Each client stores its own data and cannot read the data of other clients. Data is not independently or identically distributed.

Moreover, privacy is an indisputably crucial matter in the world of causal inference. Considering the potential of federated learning to preserve privacy and building up sophisticated models simultaneously, one might wonder whether it is possible to adopt causal learning algorithms to the federated paradigm, aiming at developing a novel and distributed generation of causal inference methods.

## Results
We design a set of experiments showcasing the superiority of our method in decentralized setup, and comparing our approach to naive voting-based predecessors. The following are just a taste of the project's final results.

<p align="center">
<img src="plots/balanced_int_rounds.png" align="center" alt="Federated Learnign per Rounds"/>
</p>

---

<p align="center">
<img src="plots/balanced_int_dataset.png" align="center" alt="Dataset Size Effect"/>
</p>

---

<p align="center">
<img src="plots/client_sweep_nodiv.png" align="center" alt="Clients Effect"/>
</p>

## Getting started

Here are some quick instructions on getting started with the repository and running federated experiments.

### General structure

To acquire a better understanding of the environment and features, you just need to clone the repository into your local machine. At first glance, the structure of the project appears below.

    ├── federated          # Source of the federated setup and experiments.
    ├── baselines          # Libraries for base-lining our federated method.
    ├── plots              # Plots produced by notebooks in federated/cluster.
    ├── environment.yml    # Required libs and packages.
    ├── LICENSE
    └── README.md

Other folders are libraries related to our local learning method and thus are not the main focus of this repository.

### Main simulation file
To run a simulation, you can use the federated simulation file inside the federated folder. The FederatedSimulator class will enable experiments with a through federated setup and two different aggregation methods. Refer to the class description for more information.

## Reproducibility
To reproduce the plots and tables in the final paper, one must run all the experiments in the federated/cluster_experiments.py with proper command structure as given by the file itself. Just remember that the local learning method of each client, ENCO, is computationally demanding especially without a GPU; therefore, running without a GPU will take more than five days on an average Core-i7 computer without a GPU.

After the successful execution of each experiment, the resulting data must be moved into the federated/cluster/data folder as appears in the repository. This folder is already filled with the previous data for ease of reproducibility. The cluster_results_er.ipynb and cluster_results_pr.ipynb notebook can reproduce the plots in total agreement to the paper.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting **pull requests** to us.

## Citation

Please use the following BibTeX formatted **citation**:
```
    @article{abyaneh2022fed-cd,
      title={FED-CD: Federated Causal Discovery from Interventional and Observational Data},
      author={Abyaneh, Amin and Scherrer, Nino and Schwab, Patrick and Bauer, Stefan and Sch{\"o}lkopf, Bernhard and Mehrjou, Arash},
      journal={arXiv preprint arXiv:2211.03846},
      year={2022}
    }
```

## Development

* [Amin Abyaneh](https://github.com/aminabyaneh) -- **maintainer**
* [Nino Scherrer](https://github.com/ninodimontalcino)
* [Arash Mehrjou](https://github.com/amehrjou)

## Authors


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

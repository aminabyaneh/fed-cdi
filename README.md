# Federated Causal Discovery From Interventions

The Causal Federated Learning project aims at designing a Cross-Silo Federated Causal Learning algorithms, capable of extracting the underlying Directed Acyclic Graph (DAG) of a distributed dataset, where each agent only has limited access to a specific set of dataset variables and a subset of intervened variables.

For further elaboration and details, you can refer to our [paper](https://arxiv.org/abs/2211.03846).

## Motivation

Existing causal discovery methods typically require the data to be available in a centralized location. However, many practical domains, such as healthcare, limit access to the data gathered by local entities, primarily for privacy and regulatory constraints. Researchers propose distributed and federated causal discovery approaches for decentralized scenarios. While methods that provide a federated perspective on causal discovery primarily emphasize observational data, the implications of utilizing interventional data remains largely unexplored.

## Method

We propose a federated framework for inferring causal structures from distributed data containing both observational and interventional data. By exchanging updates instead of samples, we ensure privacy while enabling decentralized discovery of the underlying causal graph. We accommodate scenarios with shared or disjoint intervened covariates, and mitigate the adverse effects of interventional data heterogeneity. We provide empirical evidence for the performance and scalability of our approach for decentralized causal discovery using synthetic and real-world graphs.

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

**To avoid running the method from scratch, you can download our [training data](https://drive.google.com/file/d/1W9JL4iOcUkQhXV0gfNvDpjMkmNt1Jqzf/view?usp=sharing) and simply unpack it next to the plot notebooks in the [cluster folder](federated/cluster/).**

After the successful execution of each experiment, the resulting data must be moved into the federated/cluster/data folder as appears in the repository. The cluster_results_er.ipynb and cluster_results_pr.ipynb notebook can reproduce the plots in agreement with the paper.

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


## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

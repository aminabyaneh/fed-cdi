# Federated Causal Discovery

The Causal Federated Learning project aims at designing a Cross-Silo Federated Causal Learning algorithms, capable of extracting the underlying Directed Acyclic Graph (DAG) of a distributed dataset, where each agent only has limited access to a specific set of dataset variables.

## Motivation

Federated Learning trains an algorithm across multiple decentralized devices holding local data samples, without exchanging the private information. Data is generated locally and remains de-centralised. Each client stores its own data and cannot read the data of other clients. Data is not independently or identically distributed.

Moreover, privacy is an indisputably crucial matter to the world of causal inference. Considering the potential of federated learning to preserve privacy and building up sophisticated models simultaneously, one might wonder whether it is possible to adopt causal learning algorithms to the federated paradigm, aiming at developing a novel and distributed generation of causal inference methods.

## Getting started

Here are some quick instructions about getting familiarized with Pyfectious environment and capabilities.

### General structure

To acquire a better understanding of the environment and features, you just need to clone the repository into your local machine. At first glance, the overall structure of the project appears below.

    ├── src          # Python source files of the project.
    ├── cluster      # Tools and scripts to run on the cluster.
    ├── data         # Simulator's database and configuration files.
    ├── requirements.txt
    ├── LICENSE
    └── README.md

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting **pull requests** to us.

## Citation

Please use the following BibTeX formatted **citation**:
    @article{,
      title={},
      author={},
      journal={arXiv preprint arXiv:2103.15561},
      year={2022}
    }

## Authors

* [Amin Abyaneh](https://github.com/aminabyaneh) -- **maintainer**
* [Nino Scherrer] (https://github.com/ninodimontalcino)
* [Arash Mehrjou](https://github.com/amehrjou)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

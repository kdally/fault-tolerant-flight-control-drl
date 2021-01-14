<h1 align="center">Welcome to the First  DRL Controller for a Jet Aircraft </h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-blue.svg" />
  <a href="#" target="_blank">
   <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/Django.svg" />
    <a href="#" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>


> Fault tolerant flight control for the Cessna Citation 500. First-ever use of Deep Reinforcement Learning for the flight control of a jet aircraft. Employed Soft Actor Critic (SAC). 

<p align="center">
  <img src="assets/citation_550_header.svg" width="750"/>
</p>

## Installation
> MacOS users can use the module straight away.
> Linux and Windows users are required to recompile the CitAST high-fidelity simulation model as instructed in `docs/CitAST_for_Python.pdf`.

1. Clone fault-tolerant-flight-control-drl
```sh
cd <installation_path_of_your_choice>
git clone https://github.com/kdally/fault-tolerant-flight-control-drl
cd fault-tolerant-flight-control-drl
```

2. Install required packages
 > Only compatible with TensorFlow 1.XX

```sh
python setup.py install
```

## Usage

1. To fly the aircraft right away with pre-trained controllers 🛩
```sh
python tests/test_all.py
```

> Select flight settings on the GUI. Default choices are recommended for unexperienced users.

<p align="center">
  <img src="https://i.ibb.co/fxgW2MV/GUI.png" width="350"/>
</p>


2. To retrain the inner-loop and outer-loop controllers ⚙️
```sh
python tests/train_inner_loop.py
python tests/train_outer_loop.py
```

3. To perform a hyperparameter optimization 🎯
```sh
python tests/optimization.py
```


## Author

👤 **Killian Dally**
(Delft University of Technology MSc. Student)
* Github: [@kdally](https://github.com/kdally)
* LinkedIn: [@kdally](https://linkedin.com/in/kdally)

> Project developed as part of a Master's Thesis at the Control & Simulation Division of Delft University of Technology under the supervision of Assistant Professor Dr. E-J. van Kampen.

## References

* Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. (2018) [[paper](https://arxiv.org/abs/1801.01290)][[code](https://github.com/haarnoja/sac/tree/master/sac)]
* Haarnoja, T., Zhou, A., Abbeel, P., and Hartikainen, K. (2019) [[paper](https://arxiv.org/abs/1812.05905)][[code](https://github.com/rail-berkeley/softlearning/)]
* Hill, A. et al. (2018) [[doc](https://stable-baselines.readthedocs.io/)][[code](https://github.com/hill-a/stable-baselines)]


***

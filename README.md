<h1 align="center">Welcome to the First  DRL Controller for a Jet Aircraft 🛩</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-blue.svg?cacheSeconds=2592000" />
  <a href="#" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

> Fault tolerant flight control for the Cessna Citation 500. First-ever use of Deep Reinforcement Learning for jet aircraft. Employed Soft Actor Critic (SAC). 

![](https://ibb.co/WnBTmDy)
![](https://ibb.co/1dvC4wc)

## Install

1. Clone fault-tolerant-flight-control-drl
```sh
cd <installation_path_of_your_choice>
git clone https://github.com/kdally/fault-tolerant-flight-control-drl
cd fault-tolerant-flight-control-drl
```

2. Install required packages
 > Note: only compatible with TensorFlow 1.14

```sh
pip install -r requirements.txt
```

## Usage

1. To fly the aircraft right away 🚀
```sh
python evaluate.py
```

2. To train the inner-loop and outer-loop controllers ⚙️
```sh
python train_inner_loop.py
python train_outer_loop.py
```

3. To perform a hyperparameter optimization 🎯
```sh
python optimization.py
```


## Author

👤 **Killian Dally**
(Delft University of Technology MSc. Student)
* Github: [@kdally](https://github.com/kdally)
* LinkedIn: [@kdally](https://linkedin.com/in/kdally)

> Project developed as part of a Master's Thesis at the Control & Simulation Division of Delft University of Technology.

## References

* Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S (2018) [[paper](https://arxiv.org/abs/1801.01290)][[code](https://github.com/haarnoja/sac/tree/master/sac)]
* Hill, A. et al. (2018) [[doc](https://stable-baselines.readthedocs.io/)][[code](https://github.com/hill-a/stable-baselines)]
***

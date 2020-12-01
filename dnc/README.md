# Divide-and-Conquer Reinforcement Learning

This repository contains code accompaning the paper, [Divide-and-Conquer Reinforcement Learning (Ghosh et al., ICLR 2018)](https://arxiv.org/abs/1711.09874). It includes code for the DnC algorithm, and the Mujoco environments used for the empirical evaluation. Please see [the project website](http://dibyaghosh.com/dnc/) for videos and further details.

<img src="videos/catching.gif" width="400px"/> <img src="videos/lobbing.gif" width="400px"/>

### Dependencies

This codebase requires a valid installation of `rllab`. Please refer to the [rllab repository](https://github.com/rll/rllab) for installation instructions.

The environments are built in Mujoco 1.31: follow the instructions [here](https://github.com/openai/mujoco-py/tree/0.5) to install Mujoco 1.31 if not already done. You are required to have a Mujoco license to run any of the environments.

### Usage

Sample scripts for working with DnC and the provided environments can be found in the [examples](examples/) directory. In particular, a sample scripts for running DnC is located [here](examples/dnc_pick.py).

```bash
source activate rllab_env
python examples/dnc_pick.py
```

Environments are located in the [dnc/envs/](dnc/envs/) directory, and the DnC implementation can be found at [dnc/algos/](dnc/algos).

### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/dibyaghosh/dnc/issues).

### Citing 

If you use DnC, please cite the following paper:

- Dibya Ghosh, Avi Singh, Aravind Rajeswaran, Vikash Kumar, Sergey Levine. "[Divide-and-Conquer Reinforcement Learning](https://arxiv.org/abs/1711.09874)". _Proceedings of the International Conference on Learning Representaions (ICLR), 2018._

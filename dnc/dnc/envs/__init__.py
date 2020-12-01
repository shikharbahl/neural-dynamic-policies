from dnc.envs.picker import *
from dnc.envs.throw import *
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place import *
from metaworld.envs.mujoco.sawyer_xyz.sawyer_soccer import *
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door import *
from metaworld.envs.mujoco.sawyer_xyz.sawyer_faucet_open import *
from dnc.envs.base import create_env_partitions


import numpy as np

_envs = {
    'pick_pos': PickerPosEnv,
    'throw_pos':ThrowerPosEnv,
    'push_pos': SawyerReachPushPickPlaceEnv,
    'soccer_pos': SawyerSoccerEnv,
    'faucet_pos': SawyerFaucetOpenEnv,

}

_deterministic_params = {
    'throw_pos':dict(box_center=(0,0), box_noise=0),
    'pick_pos': dict(goal_args=('noisy',(.6,.2),0)),
    'push_pos': dict(),
    'soccer_pos':dict(),
    'faucet_pos':dict(),
}

def create_stochastic(name):
    assert name in _stochastic_params
    return _envs[name](**_stochastic_params[name])

def create_deterministic(name, env_kwargs=None):
    assert name in _deterministic_params
    if env_kwargs:
        for key in env_kwargs.keys():
            _deterministic_params[name][key] = env_kwargs[key]
    return _envs[name](**_deterministic_params[name])

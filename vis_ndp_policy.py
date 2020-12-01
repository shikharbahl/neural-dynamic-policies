import copy
import glob
import os
import time
from collections import deque
from datetime import datetime
import argparse
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

from a2c_ppo_acktr import algo, utils
from arguments import get_args_ppo
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage


import cv2
args = get_ppo_args()
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
exp_id = args.expID
if args.name:
    args.save_dir = path + '/data/' + args.name + '/' + str('{:05d}'.format(exp_id)) + '_' + args.type + '_' + args.env_name + '_s'
else:
    args.save_dir = path + '/data/' + str('{:05d}'.format(exp_id)) + '_' + args.type + '_' + args.env_name + '_s'
os.environ["OPENAI_LOGDIR"] = args.save_dir + '/tmp/'
args.log_dir = args.save_dir + '/tmp/'


env_name = args.env_name + '_pos'
args.env_name = env_name
path = args.save_dir + '/actor_critic' +  '.pt'
model = torch.load(path)
actor_critic = model[0]
ob_rms = model[1]
torch.set_num_threads(1)


env_kwargs = dict()

args.num_processes = 2

env_kwargs['timestep'] = args.timestep

device = torch.device("cuda:0" if args.cuda else "cpu")
test_envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                     args.gamma, args.log_dir, device, False, env_kwargs=env_kwargs)

rollouts = RolloutStorage(args.num_steps, args.num_processes,
                          test_envs.observation_space.shape, test_envs.action_space,
                          actor_critic.recurrent_hidden_state_size)


vec_norm = utils.get_vec_normalize(test_envs)
vec_norm.eval()
vec_norm.ob_rms = ob_rms

for i in range(5):
    obs = test_envs.reset()
    for step in range(args.num_steps):
        if step % args.T == 0:
            with torch.no_grad():
                values, actions, action_log_probs_list, recurrent_hidden_states_lst = actor_critic.act(
                    obs, rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], deterministic=True)
        action = actions[step % args.T]
        obs, reward, done, infos = test_envs.step(action)
        images = test_envs.get_images()

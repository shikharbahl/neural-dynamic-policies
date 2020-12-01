
import numpy as np
from arguments import *
import copy
import glob
import os
import time
from collections import deque
from datetime import datetime
import torch.nn.functional as F
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, DMPPolicy
from a2c_ppo_acktr.storage import RolloutStorage, RolloutStorageDMP
import dnc.envs as dnc_envs
from ppo_train import train as train_ppo
from dmp_train import train as train_dmp
from a2c_ppo_acktr.algo.ppo import PPODMP


def dmp_experiment(args):
    args.l = args.num_int_steps // args.T + 1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    env_kwargs = dict(timestep=args.timestep, reward_delay=args.T)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, env_kwargs=env_kwargs)

    test_envs = make_vec_envs(args.env_name, args.seed + args.num_processes, args.num_processes,
                         None, args.log_dir, device, False, env_kwargs=env_kwargs)

    secondary_output = False

    if "throw" in args.env_name:
        state_index = np.arange(9)
        vel_index = np.arange(18, 27)

    if "pick" in args.env_name:
        state_index = np.arange(9)
        vel_index = np.arange(16, 25)

    if "push" in args.env_name:
        state_index = np.arange(3)
        vel_index = []
        env_kwargs['params'] = 'random_goal_unconstrained'

    if "soccer" in args.env_name:
        state_index = np.arange(3)
        vel_index = []
        env_kwargs['params'] = 'random_goal_unconstrained'

    if "faucet" in args.env_name:
        state_index = np.arange(3)
        vel_index = []
        secondary_output = True


    hidden_sizes = [args.hidden_size, args.hidden_size]


    actor_critic = DMPPolicy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy,
                    'hidden_size':args.hidden_size,
                    'T':args.T,
                    'N':args.N,
                    'l':args.l,
                    'goal_type':args.goal_type,
                    'hidden_sizes':hidden_sizes,
                    'state_index':state_index,
                    'vel_index': vel_index,
                    'rbf':args.rbf,
                    'a_z': args.a_z,
                    'secondary_output':secondary_output},
        )
    actor_critic.to(device)

    agent = PPODMP(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)


    rollouts = RolloutStorageDMP(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, args.T)

    train_dmp(actor_critic, agent, rollouts, envs, test_envs, args)


def ppo_experiment(args):


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if args.cuda else "cpu")


    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_kwargs = dict()

    env_kwargs['timestep'] = args.timestep

    if "push" in args.env_name:
        env_kwargs['params'] = 'random_goal_unconstrained'

    if "soccer" in args.env_name:
        env_kwargs['params'] = 'random_goal_unconstrained'

    if "faucet" in args.env_name:
        secondary_output = True


    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.gamma, args.log_dir, device, False, env_kwargs=env_kwargs)

    test_envs = make_vec_envs(args.env_name, args.seed + args.num_processes, args.num_processes,
                             None, args.log_dir, device, False, env_kwargs=env_kwargs)

    actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)



    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    train_ppo(actor_critic, agent, rollouts, envs, test_envs, args)

if __name__ == '__main__':
    args = get_args_ppo()
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    exp_id = args.expID
    if args.name:
        args.save_dir = path + '/data/' + args.name + '/' + str('{:05d}'.format(exp_id)) + '_' + args.type + '_' + args.env_name + '_s'
    else:
        args.save_dir = path + '/data/' + str('{:05d}'.format(exp_id)) + '_' + args.type + '_' + args.env_name + '_s'
    os.environ["OPENAI_LOGDIR"] = args.save_dir + '/tmp/'
    args.log_dir = args.save_dir + '/tmp/'
    args.num_env_steps = 25000 * args.num_processes * args.num_steps


    os.makedirs(args.save_dir)

    if args.type == 'dmp':
            env_name = args.env_name
            args.env_name += '_pos'
            args.goal_type = 'int_path'
            dmp_experiment(args)

    if args.type == 'ppo':
            env_name = args.env_name
            args.env_name += '_pos'
            ppo_experiment(args)

    if args.type == 'ppo-multi':
            env_name = args.env_name
            args.env_name += '_pos'
            args.goal_type = 'multi_act'
            dmp_experiment(args)

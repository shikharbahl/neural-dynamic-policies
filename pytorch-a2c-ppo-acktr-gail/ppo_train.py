import copy
import glob
import os
import time
import sys
from collections import deque
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
import dnc.envs as dnc_envs
import torch.distributions as td

def train(actor_critic, agent, rollouts, envs, test_envs, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=50)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    epoch_data = dict(distance_train_sample=[],
                      success_train_sample=[],
                      distance_train_det=[],
                      success_train_det=[],
                      distance_test_sample=[],
                      success_test_sample=[],
                      distance_test_det=[],
                      success_test_det=[])
    rollout_infos =  dict(final_distance=[], final_success_rate=[])
    args.save_interval = args.log_interval
    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            obs, reward, done, infos = envs.step(action)

            episode_rewards.append(reward[0].item())
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()
        success = [1.0*info['success'] for info in infos]
        dist = [info['distance'] for info in infos]
        rollout_infos['final_distance'].append(np.mean(dist))
        rollout_infos['final_success_rate'].append(np.mean(success))


        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            vec_norm = utils.get_vec_normalize(test_envs)
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms
            obs = test_envs.reset()
            for step in range(args.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        obs, rollouts.recurrent_hidden_states[step], rollouts.masks[step], deterministic=True)
                obs, reward, done, infos = test_envs.step(action)


            dist = [info['distance'] for info in infos]
            success = [1.0*info['success'] for info in infos]
            test_dist = np.mean(dist)
            test_sucess = np.mean(success)
            epoch_data['distance_test_det'].append(test_dist)
            epoch_data['success_test_det'].append(test_sucess)


            obs = test_envs.reset()
            for step in range(args.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        obs, rollouts.recurrent_hidden_states[step], rollouts.masks[step], deterministic=False)

                obs, reward, done, infos = test_envs.step(action)

            dist = [info['distance'] for info in infos]
            success = [1.0*info['success'] for info in infos]
            test_dist = np.mean(dist)
            test_sucess = np.mean(success)
            epoch_data['distance_test_sample'].append(test_dist)
            epoch_data['success_test_sample'].append(test_sucess)


            epoch_data['distance_train_sample'].append(np.mean(rollout_infos['final_distance']))
            epoch_data['success_train_sample'].append(np.mean(rollout_infos['final_success_rate']))
            end = time.time()
            num_epochs = j // args.log_interval
            total_num_steps = (j + 1) * args.num_processes * args.num_steps


            print( "Epochs {}, Updates {}, num timesteps {}, FPS {} \n  final distance {:0.4f}, final success {:0.2f}, final test distance {:0.4f}, final test success {:0.2f} \n"
                .format(num_epochs, j, total_num_steps, int(total_num_steps / (end - start)),
                    np.mean(rollout_infos['final_distance']), np.mean(rollout_infos['final_success_rate']), test_dist, test_sucess))
            sys.stdout.flush()

            rollout_infos =  dict(final_distance=[], final_success_rate=[])


        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = args.save_dir
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None),
                                                    getattr(utils.get_vec_normalize(envs), 'ret_rms', None)],
                                                        os.path.join(save_path, "actor_critic.pt"))
            torch.save(agent.optimizer.state_dict(), os.path.join(save_path, "agent_optimizer.pt"))
            np.save(save_path + '/epoch_data.npy', epoch_data)

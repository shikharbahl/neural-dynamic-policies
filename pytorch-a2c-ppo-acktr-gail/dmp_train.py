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
from a2c_ppo_acktr.algo.ppo import PPODMP
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, DMPPolicy
from a2c_ppo_acktr.storage import RolloutStorage, RolloutStorageDMP
import torch.distributions as td

def train(actor_critic, agent, rollouts, envs, test_envs, args):
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
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        envs.reset()
        for step in range(args.num_steps):
            if step % args.T == 0:
                with torch.no_grad():
                    values, actions, action_log_probs_list, recurrent_hidden_states_lst = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

                action = actions[step % args.T]
                action_log_probs = action_log_probs_list[step % args.T]
                recurrent_hidden_states = recurrent_hidden_states_lst[0]
                value = values[:, step % args.T].view(-1, 1)

            obs, reward, done, infos = envs.step(action)

            episode_rewards.append(reward[0].item())
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_probs, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()[:, 0].view(-1, 1)

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
                if step % args.T == 0:
                    with torch.no_grad():
                        values, actions, action_log_probs_list, recurrent_hidden_states_lst = actor_critic.act(
                            obs, rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step], deterministic=True)

                    action = actions[step % args.T]
                    action_log_probs = action_log_probs_list[step % args.T]
                    recurrent_hidden_states = recurrent_hidden_states_lst[0]
                    value = values[:, step % args.T].view(-1, 1)
                obs, reward, done, infos = test_envs.step(action)

            dist = [info['distance'] for info in infos]
            success = [1.0*info['success'] for info in infos]
            test_dist = np.mean(dist)
            test_sucess = np.mean(success)
            epoch_data['distance_test_det'].append(test_dist)
            epoch_data['success_test_det'].append(test_sucess)

            obs = test_envs.reset()
            for step in range(args.num_steps):
                if step % args.T == 0:
                    with torch.no_grad():
                        values, actions, action_log_probs_list, recurrent_hidden_states_lst = actor_critic.act(
                            obs, rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step], deterministic=False)
                    action = actions[step % args.T]
                    action_log_probs = action_log_probs_list[step % args.T]
                    recurrent_hidden_states = recurrent_hidden_states_lst[0]
                    value = values[:, step % args.T].view(-1, 1)
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

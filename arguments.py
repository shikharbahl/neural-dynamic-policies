import argparse
import torch

def get_args_ppo():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--type', default='ppo', help='dmp | ppo | ppo-multi')
    parser.add_argument('--algo', default='ppo', help='algorithm to use: ppo')
    parser.add_argument('--expID', type=int, default=1)

    ################################################### PPO args #############################################################
    parser.add_argument('--hidden-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=True, help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true',default=False, help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=10, help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=50, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int,default=10, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.1, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--log-interval', type=int, default=5, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=1000, help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None, help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--num-env-steps', type=int, default=4000000, help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--env-name', default='reach_pos', help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/', help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_true', default=False, help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=True, help='use a linear schedule on the learning rate')
    parser.add_argument('--name', default='')
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--goal-type', type=str, default="multi_act")
    parser.add_argument('--hidden-sizes', default=[100, 100])
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--timestep', type=float, default=0.01)
    parser.add_argument("--rbf", default="gaussian", type=str)
    parser.add_argument('--num-int-steps', type=int, default=35)
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--reward-delay', type=int, default=1)
    parser.add_argument('--a_z', type=float, default=25)


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()


    return args

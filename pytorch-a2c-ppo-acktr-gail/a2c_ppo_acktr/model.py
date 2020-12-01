import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dnc.envs import *
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, DiagGaussianDist
from a2c_ppo_acktr.utils import init
from dmp.utils.dmp_layer import DMPIntegrator, DMPParameters
from a2c_ppo_acktr import pytorch_util as ptu




class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

class DMPPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(DMPPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = DMPMLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)
        self.T = base_kwargs['T']
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussianDist(num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError



    def act(self, inputs, rnn_hxs, masks, deterministic=False, index=0):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        actions = []
        action_log_prob_lst = []
        for i in range(actor_features.shape[2]):
            ac_feat = actor_features[:, :, i]
            dist = self.dist(ac_feat)
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()
            actions.append(action)
            action_log_prob_lst.append(action_log_probs)
            return value, actions, action_log_prob_lst, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, ind, aab):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        value = torch.diag(value[:, ind]).view(-1, 1)

        n = actor_features.shape[2]
        actor_features = torch.stack([actor_features[i, :, ind[i]] for i in range(len(ind))])
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()


        return value, action_log_probs, dist_entropy, rnn_hxs

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class DMPNet(nn.Module):
    def __init__(
            self,
            input_size,
            init_w=3e-3,
            hidden_sizes=None,
            hidden_activation=F.tanh,
            output_activation=None,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            N = 5,
            T = 10,
            l = 10,
            az=False,
            tau = 1,
            goal_type='int_path',
            rbf='gaussian',
            only_g=False,
            a_z=25,
            state_index=np.arange(7, 16),
            vel_index = np.arange(22, 31),
            secondary_output=False,
            *args,
            **kwargs
    ):
        super().__init__()
        self.N = N
        self.l = l
        self.goal_type = goal_type
        self.vel_index = vel_index
        self.output_size = N*len(state_index) + len(state_index)
        if az:
            self.output_size += 1
        output_size = self.output_size
        dt = tau / (T*self.l)
        self.T = T
        self.output_activation=torch.tanh
        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None, a_z=a_z)
        self.only_g = only_g
        self.func = DMPIntegrator(rbf=rbf, only_g=self.only_g, az=az)
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)
        self.state_index = state_index
        self.output_dim = output_size
        self.input_size = input_size
        self.hidden_activation = hidden_activation
        self.fcs = []

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        if self.goal_type == 'multi_act':
            output_size = len(self.state_index) * self.T
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.last_fc = init_(nn.Linear(in_size, output_size))
        in_size = input_size
        self.secondary_output = secondary_output
        if self.secondary_output:
            self.last_fc_sec_out = init_(nn.Linear(output_size, self.T))
        self.dist = DiagGaussianDist(output_size)
        self.secondary_activaton = nn.Tanh()




    def forward(self, input, return_preactivations=False, first_step=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        output = self.last_fc(h)*100
        if self.goal_type == 'multi_act':
            a = output.view(output.shape[0], len(self.state_index), -1)[:, :, :self.T]
            if self.secondary_output:
                secondary = self.last_fc_sec_out(output)
                secondary = secondary.view(-1, 1, self.T)
                a = torch.cat([a, secondary], dim=1)
            return a
        if self.goal_type == 'int_path':
            y0 = input[:, self.state_index].reshape(input.shape[0]*len(self.state_index))
            if len(self.vel_index) > 0:
                dy0 = input[:, self.vel_index].reshape(input.shape[0]*len(self.state_index))
            else:
                dy0 = torch.ones_like(y0)*0.05
            y, dy, ddy = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0)
            y = y.view(input.shape[0], len(self.state_index), -1)
            y = y[:, :, ::self.l]
            a = y[:, :, 1:] - y[:, :, :-1]
            if self.secondary_output:
                secondary = self.last_fc_sec_out(output)
                secondary = secondary.view(-1, 1, self.T)
                a = torch.cat([a, secondary], dim=1)
            return a
        return a

class DMPMLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=100, **kwargs):
        super(DMPMLPBase, self).__init__(recurrent, num_inputs, hidden_size)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        T, N, l = kwargs['T'], kwargs['N'], kwargs['l']
        hidden_sizes = kwargs['hidden_sizes']
        self.actor = DMPNet(num_inputs, **kwargs)
        self.T = T
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, 3*hidden_size)), nn.Tanh(),
            init_(nn.Linear(3*hidden_size, 3*hidden_size)), nn.Tanh(),
            init_(nn.Linear(3*hidden_size, 1)), nn.Identity())

        self.output_dim = self.actor.output_dim
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        critic = self.critic(x).repeat(1, self.T)

        hidden_actor = self.actor(x)
        return critic, hidden_actor, rnn_hxs

class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=100):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(), )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

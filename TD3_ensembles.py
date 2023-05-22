import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)

		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class CriticEnsemble(nn.Module):
	def __init__(self, obs_dim, act_dim, n_critic=10):
		super().__init__()
		self.critic_ensemble = [nn.Sequential(
			nn.Linear(obs_dim + act_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
		).cuda() for _ in range(n_critic)]
		self.n_critic = n_critic

	def forward(self, obs, act):
		input = torch.cat((obs, act), dim=-1)
		outputs = [torch.squeeze(self.critic_ensemble[i](input), dim=-1) for i in range(self.n_critic)]
		return outputs


def fanin_init(tensor, scale=1):
	size = tensor.size()
	if len(size) == 2:
		fan_in = size[0]
	elif len(size) > 2:
		fan_in = np.prod(size[1:])
	else:
		raise Exception("Shape must be have dimension at least 2.")
	bound = scale / np.sqrt(fan_in)
	return tensor.data.uniform_(-bound, bound)


class ParallelizedLayerMLP(nn.Module):

	def __init__(
			self,
			ensemble_size,
			input_dim,
			output_dim,
			w_std_value=1.0,
			b_init_value=0.0
	):
		super().__init__()

		# approximation to truncated normal of 2 stds
		w_init = torch.randn((ensemble_size, input_dim, output_dim))
		w_init = torch.fmod(w_init, 2) * w_std_value
		self.W = nn.Parameter(w_init, requires_grad=True)

		# constant initialization
		b_init = torch.zeros((ensemble_size, 1, output_dim)).float()
		b_init += b_init_value
		self.b = nn.Parameter(b_init, requires_grad=True)

	def forward(self, x):
		# assumes x is 3D: (ensemble_size, batch_size, dimension)
		return x @ self.W + self.b


class ParallelizedEnsembleFlattenMLP(nn.Module):

	def __init__(
			self,
			ensemble_size,
			hidden_sizes,
			input_size,
			output_size,
			init_w=3e-3,
			hidden_init=fanin_init,
			w_scale=1,
			b_init_value=0.1,
			layer_norm=None,
			batch_norm=False,
			final_init_scale=None,
	):
		super().__init__()

		self.ensemble_size = ensemble_size
		self.input_size = input_size
		self.output_size = output_size
		self.elites = [i for i in range(self.ensemble_size)]

		self.sampler = np.random.default_rng()

		self.hidden_activation = torch.nn.ReLU()
		self.output_activation = torch.nn.Identity()

		self.layer_norm = layer_norm

		self.fcs = []

		if batch_norm:
			raise NotImplementedError

		in_size = input_size
		for i, next_size in enumerate(hidden_sizes):
			fc = ParallelizedLayerMLP(
				ensemble_size=ensemble_size,
				input_dim=in_size,
				output_dim=next_size,
			)
			for j in self.elites:
				hidden_init(fc.W[j], w_scale)
				fc.b[j].data.fill_(b_init_value)
			self.__setattr__('fc%d' % i, fc)
			self.fcs.append(fc)
			in_size = next_size

		self.last_fc = ParallelizedLayerMLP(
			ensemble_size=ensemble_size,
			input_dim=in_size,
			output_dim=output_size,
		)
		if final_init_scale is None:
			self.last_fc.W.data.uniform_(-init_w, init_w)
			self.last_fc.b.data.uniform_(-init_w, init_w)
		else:
			for j in self.elites:
				torch.nn.init.orthogonal_(self.last_fc.W[j], final_init_scale)
				self.last_fc.b[j].data.fill_(0)

	def forward(self, *inputs, **kwargs):
		flat_inputs = torch.cat(inputs, dim=-1)

		state_dim = inputs[0].shape[-1]

		dim = len(flat_inputs.shape)
		# repeat h to make amenable to parallelization
		# if dim = 3, then we probably already did this somewhere else
		# (e.g. bootstrapping in training optimization)
		if dim < 3:
			flat_inputs = flat_inputs.unsqueeze(0)
			if dim == 1:
				flat_inputs = flat_inputs.unsqueeze(0)
			flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)

		# input normalization
		h = flat_inputs

		# standard feedforward network
		for _, fc in enumerate(self.fcs):
			h = fc(h)
			h = self.hidden_activation(h)
			if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
				h = self.layer_norm(h)
		preactivation = self.last_fc(h)
		output = self.output_activation(preactivation)

		# if original dim was 1D, squeeze the extra created layer
		if dim == 1:
			output = output.squeeze(1)

		# output is (ensemble_size, batch_size, output_size)
		return output

	def sample(self, *inputs):
		preds = self.forward(*inputs)

		return torch.min(preds, dim=0)[0]
		# return torch.max(preds, dim=0)[0]

	def fit_input_stats(self, data, mask=None):
		raise NotImplementedError


class Vnet(nn.Module):
	def __init__(self, state_dim):
		super(Vnet, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, state):
		value = F.relu(self.l1(state))
		value = F.relu(self.l2(value))
		value = self.l3(value)

		return value


class TD3(object):
	def __init__(
			self,
			state_dim,
			action_dim,
			max_action,
			discount=0.99,
			tau=0.005,
			policy_noise=0.2,
			noise_clip=0.5,
			policy_freq=2,
			alpha=2.5,
			n_ensemble=5,
			config=None
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic_ensemble = ParallelizedEnsembleFlattenMLP(ensemble_size=n_ensemble,
															  hidden_sizes=[256, 256, 256],
															  input_size=state_dim + action_dim,
															  output_size=1).to(device)
		self.critic_target_ensemble = copy.deepcopy(self.critic_ensemble)
		self.critic_optimizer = torch.optim.Adam(self.critic_ensemble.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha
		self.n_ensemble = n_ensemble
		self.total_it = 0
		self.action_dim = action_dim
		self.prev_target_q_ens_min = None
		self.prev_target_q_ens_mean = None
		self.prev_critic_ensemble = copy.deepcopy(self.critic_ensemble)
		self.config = config

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer
		state, action, next_state, reward, not_done, _ = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
					torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
					self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q_ensemble = self.critic_target_ensemble(next_state, next_action)
			target_Q_ensemble_min = torch.min(target_Q_ensemble, dim=0)[0]
			target_Q = reward + not_done * self.discount * target_Q_ensemble_min

		# Get current Q estimates
		current_Q_ensemble = self.critic_ensemble(state, action)
		# target_Q = target_Q.unsqueeze(0).repeat(self.n_ensemble, 1, 1)
		critic_loss = torch.square(current_Q_ensemble - target_Q).sum(0).mean()

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			pi = self.actor(state)
			Q_ensemble = self.critic_ensemble(state, pi)
			Q_ensemble_min = torch.min(Q_ensemble, dim=0)[0]
			Q_ensemble_mean = Q_ensemble.mean(0).unsqueeze(0).repeat(self.n_ensemble, 1, 1).detach()
			div_num = torch.abs(Q_ensemble).detach().mean() + 1e-6
			uncertain = torch.sqrt(((Q_ensemble - Q_ensemble_mean) ** 2).mean(0) + 1e-6) / div_num
			actor_loss = -Q_ensemble_min.mean() / div_num + self.alpha * uncertain.mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.critic_ensemble.parameters(),
										   self.critic_target_ensemble.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			wandb.log({'Q': current_Q_ensemble.mean().item(),
					   'pi_ensembles_uncertain': uncertain.mean().item(),
					   'unc/Q': uncertain.mean().item() / current_Q_ensemble.mean().item(),
					   'critic_loss': critic_loss.item(),
					   'target_ensemble_min': target_Q_ensemble_min.mean().item()})

		return {'Q': current_Q_ensemble[0].mean().item(), 'critic_loss': critic_loss.item()}

	def imitating(self, replay_buffer, epochs=20000, batch_size=256):
		q_log, critic_loss_log, actor_loss_log = [], [], []
		for i in (range(epochs)):
			state, action, next_state, reward, not_done, next_action = replay_buffer.sample(batch_size)
			with torch.no_grad():
				target_Q_ensemble = self.critic_target_ensemble(next_state, next_action)
				target_Q = reward + not_done * self.discount * target_Q_ensemble
			current_Q_ensemble = self.critic_ensemble(state, action)
			critic_loss = torch.square(current_Q_ensemble - target_Q).sum(0).mean()
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Compute actor loss
			pi = self.actor(state)
			actor_loss = F.mse_loss(pi, action)
			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			# for i in range(self.critic.n_critic):
			for param, target_param in zip(self.critic_ensemble.parameters(),
										   self.critic_target_ensemble.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			if i % 2000 == 0:
				print(f'epoch: {i}, Q: {target_Q_ensemble.mean().item()}, '
					  f'critic_loss: {critic_loss.item()}, actor_loss: {actor_loss.item()}')
				q_log.append(current_Q_ensemble.mean().item())
				critic_loss_log.append(critic_loss.item())
				actor_loss_log.append(actor_loss.item())
		for param, target_param in zip(self.critic_ensemble.parameters(),
									   self.critic_target_ensemble.parameters()):
			target_param.data.copy_(param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(param.data)

		import matplotlib.pyplot as plt
		plt.plot(q_log)
		plt.show()
		plt.title('q')

	def save(self, filename):
		torch.save(self.critic_ensemble.state_dict(), filename + "_critic_ensemble")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	def load(self, filename):
		self.critic_ensemble.load_state_dict(torch.load(filename + "_critic_ensemble"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target_ensemble = copy.deepcopy(self.critic_ensemble)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

	def save_all(self, filename):
		torch.save(self, filename + '_all.pth')

	def load_all(self, filename):
		return torch.load(self, filename + '_all.pth')

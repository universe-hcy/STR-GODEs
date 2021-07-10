###########################
# Code adapted from Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import torch
import torch.nn as nn
import lib.utils as utils
from lib.utils import get_device

# GRU description: 
# http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
class GRU_unit(nn.Module):
	def __init__(self, latent_dim, input_dim,
		n_units = 100):
		super(GRU_unit, self).__init__()

		self.update_gate = nn.Sequential(
			nn.Linear(latent_dim * 2 + input_dim, n_units),
			nn.Tanh(),
			nn.Linear(n_units, latent_dim),
			nn.Sigmoid())
		utils.init_network_weights(self.update_gate)

		self.reset_gate = nn.Sequential(
			nn.Linear(latent_dim * 2 + input_dim, n_units),
			nn.Tanh(),
			nn.Linear(n_units, latent_dim),
			nn.Sigmoid())
		utils.init_network_weights(self.reset_gate)

		self.new_state_net = nn.Sequential(
			nn.Linear(latent_dim * 2 + input_dim, n_units),
			nn.Tanh(),
			nn.Linear(n_units, latent_dim * 2))
		utils.init_network_weights(self.new_state_net)


	def forward(self, y_mean, y_std, x):
		y_concat = torch.cat([y_mean, y_std, x], -1)

		update_gate = self.update_gate(y_concat)
		reset_gate = self.reset_gate(y_concat)
		concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)
		
		new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
		new_state_std = new_state_std.abs()

		new_y = (1-update_gate) * new_state + update_gate * y_mean
		new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

		return new_y, new_y_std

class Encoder_z0_ODE_RNN(nn.Module):
	# Code adapted from https://github.com/YuliaRubanova/latent_ode
	# Derive z0 by running ode backwards.
	# For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
	# Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
	# Continue until we get to z0
	def __init__(self, latent_dim, input_dim, z0_diffeq_solver = None,
				 z0_dim = None, GRU_update = None, n_gru_units = 100):
		
		super(Encoder_z0_ODE_RNN, self).__init__()

		if z0_dim is None:
			self.z0_dim = latent_dim
		else:
			self.z0_dim = z0_dim

		if GRU_update is None:
			self.GRU_update = GRU_unit(latent_dim, input_dim, 
				n_units = n_gru_units)
		else:
			self.GRU_update = GRU_update

		self.z0_diffeq_solver = z0_diffeq_solver
		self.latent_dim = latent_dim
		self.input_dim = input_dim

		self.transform_z0 = nn.Sequential(
			nn.Linear(latent_dim * 2, 100),
			nn.Tanh(),
			nn.Linear(100, self.z0_dim * 2),)
		utils.init_network_weights(self.transform_z0)

	def forward(self, data, time_steps, run_backwards = True):
		# data, time_steps -- observations and their time stamps
		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())
		n_timestep, n_node, n_dims = data.size()

		z0, hidden, = self.run_odernn(
			data, time_steps, run_backwards = run_backwards)

		z0 = z0.reshape(n_node, self.latent_dim)
		hidden = hidden.reshape(n_node, self.latent_dim)

		z0, hidden = utils.split_last_dim(self.transform_z0( torch.cat((z0, hidden), -1)))

		return z0, hidden

	def run_odernn(self, data, time_steps, run_backwards = True):
		n_timestep, n_node, n_dims = data.size()

		device = get_device(data)

		prev_y = torch.zeros((n_node, self.latent_dim)).to(device)
		hidden = torch.zeros((n_node, self.latent_dim)).to(device)

		prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]

		interval_length = time_steps[-1] - time_steps[0]
		minimum_step = interval_length / 12

		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps))
		if run_backwards:
			time_points_iter = reversed(time_points_iter)

		for i in time_points_iter:
			if max(prev_t - t_i - minimum_step)  < 0:
				inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i[0] - prev_t[0])

				ode_sol = prev_y + inc
				ode_sol = torch.stack((prev_y, ode_sol), 2)
			else:
				n_intermediate_tp = max(2, ((prev_t[0] - t_i[0]) / minimum_step[0]).int())

				time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
				ode_sol = self.z0_diffeq_solver(prev_y, time_points)

			yi_ode = ode_sol[:, :, -1]
			xi = data[i,:,:]

			prev_y, hidden = self.GRU_update(yi_ode, hidden, xi)

			prev_t, t_i = time_steps[i],  time_steps[i-1]

		return prev_y, hidden


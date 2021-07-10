import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

from torchdiffeq import odeint, odeint_adjoint

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver

class GODEFunc(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_index,
                 edge_attr,
                 num_relations,
                 num_bases,
                 bias=False,
                 **kwargs):

        super(GODEFunc, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))
        self.root = Param(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.basis)
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.xavier_uniform_(self.root)

    def forward(self, t_local, x, backwards = False):
        grad = self.get_ode_gradient_nn(t_local, x)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return torch.relu(self.propagate(
            self.edge_index, x=y, edge_attr=self.edge_attr, edge_norm=None))

    def sample_next_point_from_prior(self, t_local, y):
        """
        t_local: current time point
        y: value at the current time point
        """
        return self.get_ode_gradient_nn(t_local, y)

    def message(self, x_j, edge_index_j, edge_attr, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        out = torch.einsum('bi,rio->bro', x_j, w)
        out = (out * edge_attr.unsqueeze(2)).sum(dim=1)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if x is None:
            out = aggr_out + self.root
        else:
            out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)

class DiffeqSolver(nn.Module):
    # Code adapted from https://github.com/YuliaRubanova/latent_ode
    def __init__(self, ode_func:nn.Module, method:str='dopri5', odeint_rtol = 1e-4, odeint_atol = 1e-5):
        super(DiffeqSolver, self).__init__()
        self.ode_method = method
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol


    def forward(self, first_point, time_steps_to_predict, backwards = False):
        """
        # Decode the trajectory through ODE Solver
        """
        batch_size = len(time_steps_to_predict[0])
        time_steps = len(time_steps_to_predict)
        point_num = len(first_point) // batch_size
        pred_y = []
        for i in range(batch_size):
            pred_y.append(odeint(self.ode_func, first_point[i * point_num:(i + 1) * point_num,:], time_steps_to_predict[:, i],
                            rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method))
        pred_y = torch.stack(pred_y)
        pred_y = pred_y.permute(1, 0, 2, 3)
        pred_y = pred_y.reshape(time_steps, len(first_point), -1)

        pred_y = pred_y.permute(1,2,0)

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, n_traj_samples = 1):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        pred_y = pred_y.permute(1,2,0,3)
        return pred_y

class DiffeqSolver_prediction(nn.Module):
    def __init__(self, ode_func:nn.Module, gru_update, output_layer,
                 method:str='dopri5', odeint_rtol = 1e-4, odeint_atol = 1e-5):
        super(DiffeqSolver_prediction, self).__init__()
        self.ode_method = method
        self.ode_func = ode_func
        self.gru_update = gru_update

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.output_layer = output_layer
        self.global_step = 0

    def forward(self, z0, hidden, data, time_steps, backwards = False):
        """
        # Decode the trajectory through ODE Solver and ridership information
        """
        batch_size = len(time_steps[0])
        point_num = len(z0) // batch_size

        time_steps, id_time_steps = torch.sort(time_steps, dim = 0)

        hidden_dim = z0.shape[-1]
        n_timestep, num_nodes, output_dim = data.size()

        prev_y = z0
        prev_std = hidden
        xi = torch.zeros(prev_y.size()[0],
                         output_dim,
                         dtype=prev_y.dtype,
                         device=prev_y.device)
        predictions = []
        for t in range(0, len(time_steps)):
            if t == 0:
                inc = self.ode_func(time_steps[0] - 0.01, prev_y) * 0.01

                ode_sol = prev_y + inc
            else :
                time_steps_to_predict = time_steps[t - 1:t + 1, :]
                ode_sol = []
                for i in range(batch_size):
                    ode_sol.append(odeint(self.ode_func, prev_y[i * point_num:(i + 1) * point_num,:], time_steps_to_predict[:, i],
                                    rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)[-1])
                ode_sol = torch.stack(ode_sol)
                ode_sol = ode_sol.reshape(len(prev_y), -1)

            out = self.output_layer(ode_sol.reshape(-1, hidden_dim)).view(-1, point_num, output_dim)

            prev_y, prev_std = self.gru_update(ode_sol, prev_std, xi)

            predictions.append(out)

            use_truth_sequence = False
            if id_time_steps[t][0] < len(time_steps) // 2:
                use_truth_sequence = True

            if use_truth_sequence:
                xi = data[id_time_steps[t][0], :, :]
            else:
                xi = out.detach().view(-1, output_dim)

        return predictions

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, n_traj_samples = 1):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        pred_y = pred_y.permute(1,2,0,3)
        return pred_y



class Conv2dTime(nn.Conv2d):
    def __init__(self, in_channels, *args, **kwargs):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes

        Conv2d module where time gets concatenated as a feature map.
        Makes ODE func aware of the current time step.
        """
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)

def get_nonlinearity(name):
    """Helper function to get non linearity module, choose from relu/softplus/swish/lrelu"""
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'swish':
        return Swish(inplace=True)
    elif name == 'lrelu':
        return nn.LeakyReLU()

class Swish(nn.Module):
    def __init__(self, inplace=False):
        """The Swish non linearity function"""
        super().__init__()
        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


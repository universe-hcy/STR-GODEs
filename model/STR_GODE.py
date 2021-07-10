import torch
from torch import nn
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from model.ode_utils import GODEFunc, DiffeqSolver, DiffeqSolver_prediction
from model.encoder_decoder import *


class STR_GODEs(torch.nn.Module):

    def __init__(self, cfg, edge_index, edge_attr, n_gru_units=100):
        super(STR_GODEs, self).__init__()
        self.num_nodes = cfg['model']['num_nodes']
        self.num_output_dim = cfg['model']['output_dim']
        self.num_units = cfg['model']['rnn_units']
        self.num_input_dim = cfg['model']['input_dim']
        self.cfg = cfg
        self.batch_size = cfg['data']['batch_size']
        self.horizon = cfg['model']['horizon']
        self.num_relations = cfg['model'].get('num_relations', 3)
        self.num_bases = cfg['model'].get('num_bases', 3)
        self.device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.edge_index = edge_index
        self.edge_attr = edge_attr

#encoder
        self.gde_func1 = GODEFunc(self.num_units, self.num_units,
                            edge_index=self.edge_index,
                            edge_attr=self.edge_attr,
                            num_relations=self.num_relations,
                            num_bases=self.num_bases)

        z0_diffeq_solver = DiffeqSolver(self.gde_func1, "euler",
                                        odeint_rtol=1e-3, odeint_atol=1e-4)
        self.encoder_z0 = Encoder_z0_ODE_RNN(self.num_units, self.num_input_dim, z0_diffeq_solver,
                                        z0_dim=self.num_units, n_gru_units=self.num_nodes)
#decoder
        self.gde_func2 = GODEFunc(self.num_units, self.num_units,
                            edge_index=self.edge_index,
                            edge_attr=self.edge_attr,
                            num_relations=self.num_relations,
                            num_bases=self.num_bases)
        self.GRU_update = GRU_unit(self.num_units, self.num_input_dim,
                                   n_units=n_gru_units)
        self.output_layer = nn.Linear(self.num_units, self.num_output_dim)
        utils.init_network_weights(self.output_layer)
        self.diffeq_solver = DiffeqSolver_prediction(self.gde_func2, self.GRU_update, self.output_layer,
                                              'dopri5', odeint_rtol=1e-3, odeint_atol=1e-4)

    def forward(self, sequences):
        x_input = sequences.x
        y_input = sequences.y
        x_times = sequences.xtime
        y_times = sequences.ytime
#        y_times = torch.cat((sequences.xtime, sequences.ytime))

        z0, hidden = self.encoder_z0(x_input, x_times, run_backwards=True)

        data = torch.cat((x_input,y_input))
        times = torch.cat((x_times,y_times))
        predictions = self.diffeq_solver(z0, hidden, data, times)
        times, id_times = torch.sort(times, dim = 0)     #convenient for irregular prediction
        new_predictions = []
        for t in range(0, len(times)):
            if id_times[t][0] >= self.horizon:
                new_predictions.append(predictions[t])
        predictions = torch.stack(new_predictions)
        predictions = predictions.transpose(0, 1)
        return predictions

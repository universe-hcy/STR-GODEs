# part of this code are copied from DCRNN
import logging
import random
import os
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

class DataLoader(object):

    def __init__(self,
                 xs,
                 ys,
                 xtime,
                 ytime,
                 batch_size,
                 pad_with_last_sample=True,
                 shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            xtime_padding = np.repeat(xtime[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            ytime_padding = np.repeat(ytime[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            xtime = np.concatenate([xtime, xtime_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            ytime = np.concatenate([ytime, ytime_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
            xtime, ytime = xtime[permutation], ytime[permutation]
        self.xs = xs
        self.ys = ys
        self.xtime = xtime
        self.ytime = ytime

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size,
                              self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                xtime_i = self.xtime[start_ind:end_ind, ...]
                ytime_i = self.ytime[start_ind:end_ind, ...]
                yield (x_i, y_i, xtime_i, ytime_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler_Torch:
    """
    Standard the input
    """

    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir,
               name,
               log_filename='info.log',
               level=logging.INFO,
               write_to_file=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    if write_to_file is True:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger




def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][...,
                                                 0].mean(),
                            std=data['x_train'][...,
                                                0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][...,
                              0] = scaler.transform(data['x_' + category][...,
                                                                          0])
        data['y_' + category][...,
                              0] = scaler.transform(data['y_' + category][...,
                                                                          0])
    data['train_loader'] = DataLoader(data['x_train'],
                                      data['y_train'],
                                      batch_size,
                                      shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'],
                                    data['y_val'],
                                    test_batch_size,
                                    shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'],
                                     data['y_test'],
                                     test_batch_size,
                                     shuffle=False)
    data['scaler'] = scaler

    return data


def load_dataset_hz(dataset_dir,
                    batch_size,
                    test_batch_size=None,
                    scaler_axis=(0,
                                 1,
                                 2,
                                 3),
                    **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = load_pickle(os.path.join(dataset_dir, category + '.pkl'))
        data['x_' + category] = cat_data['x']
        data['xtime_' + category] = cat_data['xtime']
        data['y_' + category] = cat_data['y']
        data['ytime_' + category] = cat_data['ytime']
    scaler = StandardScaler(mean=data['x_train'].mean(axis=scaler_axis),
                            std=data['x_train'].std(axis=scaler_axis))
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])
    data['train_loader'] = DataLoader(data['x_train'],
                                      data['y_train'],
                                      data['xtime_train'],
                                      data['ytime_train'],
                                      batch_size,
                                      shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'],
                                    data['y_val'],
                                    data['xtime_val'],
                                    data['ytime_val'],
                                    test_batch_size,
                                    shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'],
                                     data['y_test'],
                                     data['xtime_test'],
                                     data['ytime_test'],
                                     test_batch_size,
                                     shuffle=False)
    data['scaler'] = scaler

    return data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_graph_data_hz(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    return adj_mx.astype(np.float32)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class SimpleBatch(list):

    def to(self, device):
        for ele in self:
            ele.to(device)
        return self


def collate_wrapper(x, y, edge_index, edge_attr, device, return_y=True, flip = False):
    x = torch.tensor(x, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.float, device=device)
    x = x.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    y_T_first = y.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    #  do not tranpose y_truth
    T = x.size()[0]
    N = x.size()[1]

    # generate batched sequence.
    id_seq = [i for i in range(2 * T)]
    if flip == True:
        for i in range(1, 2 * T):
            tmp = random.randint(1, i)
            id_seq[tmp], id_seq[i] = id_seq[i], id_seq[tmp]
    id_seq_x = sorted(id_seq[ : T])
    id_seq_y = sorted(id_seq[T : ])
    id_seq = id_seq_x + id_seq_y

    sequences = []
    for t in range(T):
        cur_batch_x = x[id_seq_x[t]] if id_seq_x[t] < T else y_T_first[id_seq_x[t] - T]
        cur_batch_y = x[id_seq_y[t]] if id_seq_y[t] < T else y_T_first[id_seq_y[t] - T]

        batch = Batch.from_data_list([
            Data(x=cur_batch_x[i],
                 edge_index=edge_index,
                 edge_attr=edge_attr,
                 y=cur_batch_y[i]) for i in range(N)
        ])
        sequences.append(batch)
    sequences.append(id_seq)
    if return_y:
        return SimpleBatch(sequences), y
    else:
        return SimpleBatch(sequences)

def time_trans(time_sequences):
    t_seq = []
    for _time in time_sequences:
        split_time = str(_time).split(':')
        t_seq.append(float(int(split_time[0][-2:]) * 60 + int(split_time[1])) / 15)
    return torch.Tensor(t_seq)

def collate_wrapper2(x, y, edge_index, edge_attr, device, return_y=True, xtime=None, ytime=None, irregular = False):
    x = torch.tensor(x, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.float, device=device)
    x = x.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    y_T_first = y.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    T = x.size()[0]
    N = x.size()[1]

    # generate batched sequence.
    id_seq = [i for i in range(2 * T)]
    # whether to make irregular prediction
    if irregular == True:
        for i in range(1, 2 * T):
            tmp = random.randint(1, i)
            id_seq[tmp], id_seq[i] = id_seq[i], id_seq[tmp]
    id_seq_x = sorted(id_seq[ : T])
    id_seq_y = sorted(id_seq[T : ])

    horizon, batch_size, num_nodes, num_input_dim = x.shape
    x = x.reshape(horizon, batch_size * num_nodes, num_input_dim)
    y_T_first = y_T_first.reshape(horizon, batch_size * num_nodes, num_input_dim)
    data_xtime = []
    data_ytime = []
    data_x = []
    data_y = []
    for t in range(T):
        x_times = [xtime[i][id_seq_x[t]] if id_seq_x[t] < T else ytime[i][id_seq_x[t] - T] for i in range(N)]
        y_times = [xtime[i][id_seq_y[t]] if id_seq_y[t] < T else ytime[i][id_seq_y[t] - T] for i in range(N)]

        data_xtime.append(time_trans(x_times))
        data_ytime.append(time_trans(y_times))

        data_x.append(x[id_seq_x[t]] if id_seq_x[t] < T else y_T_first[id_seq_x[t] - T])
        data_y.append(x[id_seq_y[t]] if id_seq_y[t] < T else y_T_first[id_seq_y[t] - T])
    data_xtime = torch.stack(data_xtime)
    data_ytime = torch.stack(data_ytime)
    data_x = torch.stack(data_x)
    data_y = torch.stack(data_y)

    sequences = Batch.from_data_list([
            Data(x=data_x,
                 edge_index=edge_index,
                 edge_attr=edge_attr,
                 y=data_y,
                 xtime=data_xtime,
                 ytime=data_ytime)
        ]).to(device)
    if return_y:
        return sequences, y
    else:
        return sequences

def collate_wrapper_multi_branches(x_numpy, y_numpy, edge_index_list, device):
    sequences_multi_branches = []
    for edge_index in edge_index_list:
        sequences, y = collate_wrapper(x_numpy, y_numpy, edge_index, device, return_y=True)
        sequences_multi_branches.append(sequences)

    return sequences_multi_branches, y


def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim//2

    if len(data.size()) == 3:
        res = data[:,:,:last_dim], data[:,:,last_dim:]

    if len(data.size()) == 2:
        res = data[:,:last_dim], data[:,last_dim:]
    return res

def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert(start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                torch.linspace(start[i], end[i], n_points)),0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res

def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)

    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()

def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

import torch
import torch.nn as nn
import numpy as np

from torch.distributions.categorical import Categorical


class Controller201(nn.Module):
    # we refer to https://github.com/TDeVries/enas_pytorch/blob/master/models/controller.py
    def __init__(
        self,
        nasbench,
        lstm_size=32,
        lstm_num_layers=2,
        tanh_constant=2.5,
        temperature=5.0,
    ):
        super(Controller201, self).__init__()
        # assign the attributes
        self.nasbench = nasbench
        self.max_nodes = nasbench.max_nodes
        self.num_edge = len(nasbench.edge2index)
        self.edge2index = nasbench.edge2index
        self.num_ops = len(nasbench.op_names)
        self.op_names = nasbench.op_names
        self.lstm_size = lstm_size
        self.lstm_N = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        # create parameters
        self.register_parameter(
            "input_vars", nn.Parameter(torch.Tensor(1, 1, lstm_size))
        )
        self.w_lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.lstm_N,
        )
        self.w_embd = nn.Embedding(self.num_ops, self.lstm_size)
        self.w_pred = nn.Linear(self.lstm_size, self.num_ops)

        nn.init.uniform_(self.input_vars, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_embd.weight, -0.1, 0.1)
        nn.init.uniform_(self.w_pred.weight, -0.1, 0.1)

    def get_prob(self, actions_index):
        inputs, h0 = self.input_vars, None
        log_probs, entropys, sampled_arch = [], [], []
        probs = []

        x_probs = []
        for iedge in range(self.num_edge):
            outputs, h0 = self.w_lstm(inputs, h0)

            logits = self.w_pred(outputs)

            logits = logits / self.temperature

            logits = self.tanh_constant * torch.tanh(logits)

            x_prob = torch.softmax(logits, dim=-1)
            x_probs.append(np.round(x_prob.view(-1).tolist(), 2))

            # distribution
            op_distribution = Categorical(logits=logits)
            # op_index = op_distribution.sample()
            # sampled_arch.append(op_index.item())

            op_index = actions_index[iedge]
            op_index = op_index.unsqueeze(dim=-1).unsqueeze(dim=-1)  # op_index -> [[op_index]]
            sampled_arch.append(op_index.item())

            op_log_prob = op_distribution.log_prob(op_index)

            op_prob = op_distribution.probs.squeeze()[op_index.item()].unsqueeze(-1).unsqueeze(-1)
            probs.append(op_prob.view(-1))

            log_probs.append(op_log_prob.view(-1))
            op_entropy = op_distribution.entropy()
            entropys.append(op_entropy.view(-1))

            # obtain the input embedding for the next step
            inputs = self.w_embd(op_index)

        sampled_arch = self.nasbench.convert_structure(sampled_arch)
        return (
            x_probs,
            torch.sum(torch.cat(log_probs)),
            torch.sum(torch.cat(entropys)),
            sampled_arch,
        )

    def forward(self):
        inputs, h0 = self.input_vars, None
        log_probs, entropys, sampled_arch = [], [], []
        probs = []

        x_probs = []

        for iedge in range(self.num_edge):
            outputs, h0 = self.w_lstm(inputs, h0)

            logits = self.w_pred(outputs)

            logits = logits / self.temperature

            logits = self.tanh_constant * torch.tanh(logits)

            x_prob = torch.softmax(logits, dim=-1)

            x_probs.append(np.round(x_prob.view(-1).tolist(), 2))

            # distribution
            op_distribution = Categorical(logits=logits)

            op_index = op_distribution.sample()
            sampled_arch.append(op_index.item())

            op_log_prob = op_distribution.log_prob(op_index)

            op_prob = op_distribution.probs.squeeze()[op_index.item()].unsqueeze(-1).unsqueeze(-1)
            probs.append(op_prob.view(-1))

            log_probs.append(op_log_prob.view(-1))
            op_entropy = op_distribution.entropy()
            entropys.append(op_entropy.view(-1))

            # obtain the input embedding for the next step
            inputs = self.w_embd(op_index)

        actions = sampled_arch
        sampled_arch = self.nasbench.convert_structure(sampled_arch)
        return (
            x_probs,
            torch.sum(torch.cat(log_probs)),
            torch.sum(torch.cat(entropys)),
            sampled_arch,
            actions
        )



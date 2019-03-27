# -*- coding: utf-8 -*-

import logging
import numpy as np

import torch
import torch.nn as nn

from util import *


class LSTM(nn.Module):
    def __init__(self, input_size, output_size,
                 activation='tanh', inner_activation='sigmoid', return_sequences=False):

        super(LSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.activation = parse_activation(activation)
        self.inner_activation = parse_activation(inner_activation)
        self.return_sequences = return_sequences

        # initialize model parameters
        self.w_input = nn.Parameter(torch.FloatTensor(np.zeros((input_size, output_size))))
        self.w_forget = nn.Parameter(torch.FloatTensor(np.zeros((input_size, output_size))))
        self.w_cell = nn.Parameter(torch.FloatTensor(np.zeros((input_size, output_size))))
        self.w_out = nn.Parameter(torch.FloatTensor(np.zeros((input_size, output_size))))

        self.u_input = nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.u_forget = nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.u_cell = nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.u_out = nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))

        # bias term
        self.b_input = nn.Parameter(torch.FloatTensor(np.zeros(output_size)))
        self.b_forget = nn.Parameter(torch.FloatTensor(np.zeros(output_size)))
        self.b_cell = nn.Parameter(torch.FloatTensor(np.zeros(output_size)))
        self.b_out = nn.Parameter(torch.FloatTensor(np.zeros(output_size)))

        # initialize weights
        for name, param in self.named_parameters():
            if '_b' in name or 'b_' in name:
                nn.init.constant_(param, 0.0)
            elif 'w_' in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.orthogonal_(param)

    def _step(self,
              xi_t, xf_t, xo_t, xc_t, mask_t,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c, b_u):

        i_t = self.activation(xi_t + torch.matmul(h_tm1 * b_u[0], u_i))
        f_t = self.inner_activation(xf_t + torch.matmul(h_tm1 * b_u[1], u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + torch.matmul(h_tm1 * b_u[2], u_c))
        o_t = self.inner_activation(xo_t + torch.matmul(h_tm1 * b_u[3], u_o))
        h_t = o_t * self.activation(c_t)

        h_t = (1.0 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1.0 - mask_t) * c_tm1 + mask_t * c_t

        return h_t, c_t

    def forward(self, X, mask=None, init_state=None, dropout=0, train=True, srng=None):
        # switch mode
        if train:
            self.train()
        else:
            self.eval()

        if mask is None:
            mask = torch.ones((X.shape[0], X.shape[1]))

        mask = mask.unsqueeze(2).permute(1, 0, 2).float()  # (time, nb_samples, 1)

        X = X.permute(1, 0, 2)

        retain_prob = 1. - dropout
        B_w = torch.ones((4,)).float()
        B_u = torch.ones((4,)).float()

        if dropout > 0:
            # logging.info('applying dropout with p = %f', dropout)
            if train:
                B_w = torch.from_numpy(np.random.binomial(n=1, p=retain_prob, size=(4, X.shape[1], self.input_size))).float()
                B_u = torch.from_numpy(np.random.binomial(n=1, p=retain_prob, size=(4, X.shape[1], self.input_size))).float()
            else:
                B_w *= retain_prob
                B_u *= retain_prob

        x_input = torch.matmul(X * B_w[0], self.w_input) + self.b_input
        x_forget = torch.matmul(X * B_w[1], self.w_forget) + self.b_forget
        x_cell = torch.matmul(X * B_w[2], self.w_cell) + self.b_cell
        x_output = torch.matmul(X * B_w[3], self.w_out) + self.b_out

        if init_state:
            # (batch_size, output_dim)
            first_state = init_state
        else:
            first_state = torch.zeros(X.size()[1], self.output_size)

        first_input = torch.zeros(X.size()[1], self.output_size)

        for i in range(X.size()[0]):
            # assign each timestep

            outputs, memories = self._step(x_input, x_forget, x_output, x_cell, mask, first_state, first_input,
                                           self.u_input, self.u_forget, self.u_out, self.u_cell, B_u)

            first_state = outputs
            first_input = memories

        if self.return_sequences:
            return outputs.permute(1, 0, 2)
        return outputs[-1]


class BiLSTM(nn.Module):
    def __init__(self, input_size, output_size,
                 activation='tanh', inner_activation='sigmoid', return_sequences=False):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.return_sequences = return_sequences

        params = dict(locals())
        del params['self']

        self.forward_lstm = LSTM(**params)
        self.backward_lstm = LSTM(**params)

    def forward(self, X, mask=None, init_state=None, dropout=0, train=True):
        # X: (nb_samples, nb_time_steps, embed_dim)
        # mask: (nb_samples, nb_time_steps)
        if mask is None:
            mask = torch.ones((X.shape[0], X.shape[1]))

        hidden_states_forward = self.forward_lstm(X, mask, init_state, dropout, train)
        hidden_states_backward = self.backward_lstm(torch.flip(X, dims=[1]), torch.flip(mask, dims=[1]), init_state, dropout, train)

        if self.return_sequences:
            hidden_states = torch.cat([hidden_states_forward, torch.flip(hidden_states_backward, dims=[1])], dim=-1)
        else:
            raise NotImplementedError()

        return hidden_states




from lang.grammar import Grammar
from parse import *
from astnode import *

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import config

import pdb


class PointerNet(nn.Module):
    def __init__(self):
        super(PointerNet, self).__init__()

        # three linear layers
        self.dense1_input = nn.Linear(config.encoder_hidden_dim, config.ptrnet_hidden_dim)
        self.dense1_h = nn.Linear(config.decoder_hidden_dim + config.encoder_hidden_dim, config.ptrnet_hidden_dim)
        self.dense2 = nn.Linear(config.ptrnet_hidden_dim, 1)

    def forward(self, query_embed, query_token_embed_mask, decoder_states):
        query_embed_trans = self.dense1_input(query_embed)
        h_trans = self.dense1_h(decoder_states)

        query_embed_trans = query_embed_trans.unsqueeze(1)

        h_trans = h_trans.unsqueeze(2)

        # (batch_size, max_decode_step, query_token_num, ptr_net_hidden_dim)
        dense1_trans = F.tanh(query_embed_trans + h_trans)

        scores = self.dense2(dense1_trans)
        scores = scores.reshape(scores.shape[0], scores.shape[1], -1)

        scores = torch.exp(scores - torch.max(scores, dim=-1, keepdim=True)[0])

        scores *= query_token_embed_mask.unsqueeze(1).float()
        scores = scores / torch.sum(scores, dim=-1, keepdim=True)

        return scores


class CondAttLSTM(nn.Module):
    # constructor
    def __init__(self, input_size, output_size, context_size, att_size, dropout=0.0, name='CondAttLSTM'):
        super(CondAttLSTM, self).__init__()
        self.output_size = output_size
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.context_size = context_size
        self.input_size = input_size
        # TODO: dropout
        self.dropout_layer = nn.Dropout(p=dropout)

        # initialize model parameters
        self.w_input = torch.nn.Parameter(torch.FloatTensor(np.zeros((input_size, output_size))))
        self.w_forget = torch.nn.Parameter(torch.FloatTensor(np.zeros((input_size, output_size))))
        self.w_cell = torch.nn.Parameter(torch.FloatTensor(np.zeros((input_size, output_size))))
        self.w_out = torch.nn.Parameter(torch.FloatTensor(np.zeros((input_size, output_size))))

        self.u_input = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.u_forget = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.u_cell = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.u_out = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))

        # context term
        self.c_input = torch.nn.Parameter(torch.FloatTensor(np.zeros((context_size, output_size))))
        self.c_forget = torch.nn.Parameter(torch.FloatTensor(np.zeros((context_size, output_size))))
        self.c_cell = torch.nn.Parameter(torch.FloatTensor(np.zeros((context_size, output_size))))
        self.c_out = torch.nn.Parameter(torch.FloatTensor(np.zeros((context_size, output_size))))

        # attention over tree history
        # this was not used in the original paper
        self.h_input = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.h_forget = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.h_cell = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.h_out = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))

        self.p_input = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.p_forget = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.p_cell = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))
        self.p_out = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, output_size))))

        # bias term
        self.b_input = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size))))
        self.b_forget = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size))))
        self.b_cell = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size))))
        self.b_out = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size))))

        # attention weights
        self.w_context_att = torch.nn.Parameter(torch.FloatTensor(np.zeros((context_size, att_size))))
        self.w_context_att_h = torch.nn.Parameter(torch.FloatTensor(np.zeros((output_size, att_size))))
        self.context_att_b = torch.nn.Parameter(torch.FloatTensor(np.zeros((att_size))))

        self.w_att = torch.nn.Parameter(torch.FloatTensor(np.zeros((att_size, 1))))
        self.b_att = torch.nn.Parameter(torch.FloatTensor(np.zeros((1))))

        # initialize weights
        for name, param in self.named_parameters():
            if '_b' in name or 'b_' in name:
                nn.init.constant_(param, 0.0)
            elif 'w_' in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.orthogonal_(param)

    def forward(self, X, context, parent_t_seq, init_state=None, init_cell=None, hist_h=None, mask=None,
                context_mask=None, dropout=0, train=True, srng=None, time_steps=None):

        # switch mode
        if train:
            self.train()
        else:
            self.eval()

        if mask is None:
            mask = torch.ones((X.shape[0], X.shape[1]))

        mask = mask.unsqueeze(2).permute(1, 0, 2).float()  # (time, nb_samples, 1)

        # (time, batch, hidden_size)
        X = X.permute(1, 0, 2)

        # initialize states and cells
        if init_state is None:
            init_state = torch.FloatTensor(np.zeros((X.shape[1], self.output_size)))
        if init_cell is None:
            first_cell = torch.FloatTensor(np.zeros((X.shape[1], self.output_size)))
        else:
            first_cell = init_cell
        if hist_h is None:
            hist_h = torch.FloatTensor(np.zeros((X.shape[1], X.shape[0], self.output_size)))

        x_input = torch.add(torch.matmul(X, self.w_input), self.b_input)
        x_forget = torch.add(torch.matmul(X, self.w_forget), self.b_forget)
        x_cell = torch.add(torch.matmul(X, self.w_cell), self.b_cell)
        x_out = torch.add(torch.matmul(X, self.w_out), self.b_out)

        # (batch_size, context_size, att_layer1_dim)
        context_att_tmp = torch.matmul(context, self.w_context_att) + self.context_att_b

        # before for loop
        # swap parent_t_seq dimensions
        parent_t_seq = parent_t_seq.t()

        outputs = None
        cells = None
        ctx_vectors = None

        flag = False

        # for loop through time steps
        for t in range(X.shape[0]):
            # assign each timestep
            xi_t = x_input[t]
            xf_t = x_forget[t]
            xc_t = x_cell[t]
            xo_t = x_out[t]
            mask_t = mask[t]

            parent_t = parent_t_seq[t]
            h_tm1_att_trans = torch.matmul(init_state, self.w_context_att_h)

            # (batch_size, context_size, att_layer1_dim)
            att_hidden = self.tanh(torch.add(context_att_tmp, h_tm1_att_trans[:, None, :]))
            # (batch_size, context_size, 1)
            att_raw = (torch.matmul(att_hidden, self.w_att) + self.b_att).squeeze(1)
            att_raw = att_raw.reshape(att_raw.shape[0], att_raw.shape[1])

            # (batch_size, context_size)
            ctx_att = torch.exp(att_raw - torch.max(att_raw, dim=-1, keepdim=True)[0])

            if context_mask is not None:
                ctx_att = ctx_att * context_mask.float()

            ctx_att = ctx_att / torch.sum(ctx_att, dim=-1, keepdim=True)
            # (batch_size, context_dim)
            ctx_vec = torch.sum(context * ctx_att[:, :, None], dim=1)

            # parent hidden state continuation
            if t == 0:
                par_h = torch.FloatTensor(np.zeros(init_state.shape))
            else:
                par_h = hist_h[torch.arange(hist_h.shape[0]), parent_t, :]

            # conditioned lstm
            i_t = self.sigmoid(
                xi_t + torch.matmul(init_state, self.u_input) + torch.matmul(ctx_vec, self.c_input) + torch.matmul(
                    par_h, self.p_input))
            f_t = self.sigmoid(
                xf_t + torch.matmul(init_state, self.u_forget) + torch.matmul(ctx_vec, self.c_forget) + torch.matmul(
                    par_h, self.p_forget))
            c_t = f_t * first_cell + i_t * self.tanh(
                xc_t + torch.matmul(init_state, self.u_cell) + torch.matmul(ctx_vec, self.c_cell) + torch.matmul(par_h,
                                                                                                                 self.p_cell))
            o_t = self.sigmoid(
                xo_t + torch.matmul(init_state, self.u_out) + torch.matmul(ctx_vec, self.c_out) + torch.matmul(par_h,
                                                                                                               self.p_out))
            # final output
            h_t = o_t * self.tanh(c_t)

            # mask
            h_t = (1 - mask_t) * init_state + mask_t * h_t
            c_t = (1 - mask_t) * first_cell + mask_t * c_t

            # new hist_h
            # print(hist_h.shape, h_t.shape)
            hist_h[:, t, :] = h_t

            # assign new values
            init_state = h_t
            first_cell = c_t

            # attach final result
            if flag:
                outputs = torch.cat((outputs, init_state.reshape(1, *init_state.shape)), dim=0)
                cells = torch.cat((cells, first_cell.reshape(1, *first_cell.shape)), dim=0)
                ctx_vectors = torch.cat((ctx_vectors, ctx_vec.reshape(1, *ctx_vec.shape)), dim=0)
            else:
                outputs = init_state.reshape(1, *init_state.shape)
                cells = first_cell.reshape(1, *first_cell.shape)
                ctx_vectors = ctx_vec.reshape(1, *ctx_vec.shape)
                flag = True

        # back to batch seq and size
        return outputs.squeeze(2).permute(1, 0, 2), cells.squeeze(2).permute(1, 0, 2), ctx_vectors.squeeze(2).permute(1, 0, 2)


class Hyp(object):
    def __init__(self, *args):
        if isinstance(args[0], Hyp):
            hyp = args[0]
            self.grammar = hyp.grammar
            self.tree = hyp.tree.copy()
            self.t = hyp.t
            self.hist_h = list(hyp.hist_h)
            self.log = hyp.log
            self.has_grammar_error = hyp.has_grammar_error
        else:
            assert isinstance(args[0], Grammar)
            grammar = args[0]
            self.grammar = grammar
            self.tree = DecodeTree(grammar.root_node.type)
            self.t = -1
            self.hist_h = []
            self.log = ''
            self.has_grammar_error = False

        self.score = 0.0

        self.__frontier_nt = self.tree
        self.__frontier_nt_t = -1

    def __repr__(self):
        return self.tree.__repr__()

    def can_expand(self, node):
        if self.grammar.is_value_node(node):
            # if the node is finished
            if node.value is not None and node.value.endswith('<eos>'):
                return False
            return True
        elif self.grammar.is_terminal(node):
            return False

        # elif node.type == 'epsilon':
        #     return False
        # elif is_terminal_ast_type(node.type):
        #     return False

        # if node.type == 'root':
        #     return True
        # elif inspect.isclass(node.type) and issubclass(node.type, ast.AST) and not is_terminal_ast_type(node.type):
        #     return True
        # elif node.holds_value and not node.label.endswith('<eos>'):
        #     return True

        return True

    def apply_rule(self, rule, nt=None):
        if nt is None:
            nt = self.frontier_nt()

        # assert rule.parent.type == nt.type
        if rule.parent.type != nt.type:
            self.has_grammar_error = True

        self.t += 1
        # set the time step when the rule leading by this nt is applied
        nt.t = self.t
        # record the ApplyRule action that is used to expand the current node
        nt.applied_rule = rule

        for child_node in rule.children:
            child = DecodeTree(child_node.type, child_node.label, child_node.value)
            # if is_builtin_type(rule.parent.type):
            #     child.label = None
            #     child.holds_value = True

            nt.add_child(child)

    def append_token(self, token, nt=None):
        if nt is None:
            nt = self.frontier_nt()

        self.t += 1

        if nt.value is None:
            # this terminal node is empty
            nt.t = self.t
            nt.value = token
        else:
            nt.value += token

    def frontier_nt_helper(self, node):
        if node.is_leaf:
            if self.can_expand(node):
                return node
            else:
                return None

        for child in node.children:
            result = self.frontier_nt_helper(child)
            if result:
                return result

        return None

    def frontier_nt(self):
        if self.__frontier_nt_t == self.t:
            return self.__frontier_nt
        else:
            _frontier_nt = self.frontier_nt_helper(self.tree)
            self.__frontier_nt = _frontier_nt
            self.__frontier_nt_t = self.t

            return _frontier_nt

    def get_action_parent_t(self):
        """
        get the time step when the parent of the current
        action was generated
        WARNING: 0 will be returned if parent if None
        """
        nt = self.frontier_nt()

        # if nt is a non-finishing leaf
        # if nt.holds_value:
        #     return nt.t

        if nt.parent:
            return nt.parent.t
        else:
            return 0

    # def get_action_parent_tree(self):
    #     """
    #     get the parent tree
    #     """
    #     nt = self.frontier_nt()
    #
    #     # if nt is a non-finishing leaf
    #     if nt.holds_value:
    #         return nt
    #
    #     if nt.parent:
    #         return nt.parent
    #     else:
    #         return None

import torch
import torch.nn as nn

from components import CondAttLSTM


class BaseDecoder(nn.Module):
    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward(self, inputs):
        raise NotImplemented


class LSTMDecoder(BaseDecoder):
    def __init__(self, hidden_dim, rule_embed_dim, node_embed_dim, encoder_hidden_dim, attention_hidden_dim,
                 dropout=0.2, frontier_node_type_feed=False, parent_action_feed=False):
        super(LSTMDecoder, self).__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.frontier_node_type_feed = frontier_node_type_feed
        self.parent_action_feed = parent_action_feed

        self.decoder = CondAttLSTM(rule_embed_dim + node_embed_dim + rule_embed_dim,
                                   hidden_dim, encoder_hidden_dim, attention_hidden_dim)

        # decoder_hidden_dim -> action embed
        self.hidden_state_W_rule = nn.Linear(hidden_dim, rule_embed_dim)

        # decoder_hidden_dim -> action embed
        self.hidden_state_W_token = nn.Linear(hidden_dim + encoder_hidden_dim, rule_embed_dim)

        self.hidden = None
        self.ctx_vectors = None

    def forward(self, inputs):
        decoder_input, query_embed, query_token_embed_mask, tgt_action_seq_mask, tgt_par_t_seq = inputs
        # decoder_hidden_states: (batch_size, max_example_action_num, lstm_hidden_state)
        # ctx_vectors: (batch_size, max_example_action_num, encoder_hidden_dim)
        self.hidden, _, self.ctx_vectors = self.decoder(decoder_input,
                                                        context=query_embed,
                                                        context_mask=query_token_embed_mask,
                                                        mask=tgt_action_seq_mask,
                                                        parent_t_seq=tgt_par_t_seq,
                                                        dropout=self.dropout)

        # apply additional non-linearity transformation before predicting actions
        decoder_hidden_state_trans_rule = self.hidden_state_W_rule(self.hidden)
        decoder_hidden_state_trans_token = self.hidden_state_W_token(torch.cat((self.hidden, self.ctx_vectors), dim=-1))

        return decoder_hidden_state_trans_rule, decoder_hidden_state_trans_token

    def next_step(self, inputs):
        [node_embed, par_rule_embed, prev_action_embed, time_steps,
         parent_t, decoder_prev_state, decoder_prev_cell,
         hist_h, query_embed, query_token_embed_mask] = inputs

        # (batch_size, 1, decoder_state_dim)
        # prev_action_embed_reshaped = prev_action_embed.dimshuffle((0, 'x', 1))
        prev_action_embed_reshaped = prev_action_embed.unsqueeze(1)

        # (batch_size, 1, node_embed_dim)
        node_embed_reshaped = node_embed.unsqueeze(1)

        # (batch_size, 1, node_embed_dim)
        par_rule_embed_reshaped = par_rule_embed.unsqueeze(1)

        if not self.frontier_node_type_feed:
            node_embed_reshaped *= 0.

        if not self.parent_action_feed:
            par_rule_embed_reshaped *= 0.

        decoder_input = torch.cat(
            [prev_action_embed_reshaped, node_embed_reshaped, par_rule_embed_reshaped], dim=-1)

        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, field_token_encode_dim)
        decoder_next_state_dim3, decoder_next_cell_dim3, self.ctx_vectors = self.decoder(decoder_input,
                                                                                         init_state=decoder_prev_state,
                                                                                         init_cell=decoder_prev_cell,
                                                                                         hist_h=hist_h,
                                                                                         context=query_embed,
                                                                                         context_mask=query_token_embed_mask,
                                                                                         parent_t_seq=parent_t,
                                                                                         dropout=self.dropout,
                                                                                         train=False,
                                                                                         time_steps=time_steps)

        decoder_next_state = decoder_next_state_dim3.flatten(1)
        # decoder_output = decoder_next_state * (1 - DECODER_DROPOUT)

        decoder_next_cell = decoder_next_cell_dim3.reshape(decoder_next_cell_dim3.shape[0], -1)

        decoder_next_state_trans_rule = self.hidden_state_W_rule(decoder_next_state)
        decoder_next_state_trans_token = self.hidden_state_W_token(
            torch.cat([decoder_next_state, self.ctx_vectors.flatten(1)], dim=-1))

        return decoder_next_state, decoder_next_cell, decoder_next_state_trans_rule, decoder_next_state_trans_token








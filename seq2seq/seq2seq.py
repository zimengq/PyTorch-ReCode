import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from util import get_mask, tensor_right_shift
from components import Hyp, PointerNet


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, decoder_hidden_dim, rule_num, rule_embed_dim,
                 node_num, node_embed_dim, target_vocab_size, max_query_length, head_nt_constraint,
                 output_dim=2, dropout=0.2, frontier_node_type_feed=False, parent_action_feed=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.rule_num = rule_num
        self.rule_embed_dim = rule_embed_dim
        self.node_num = node_num
        self.node_embed_dim = node_embed_dim

        self.max_query_length = max_query_length
        self.head_nt_constraint = head_nt_constraint
        self.dropout = dropout
        self.frontier_node_type_feed = frontier_node_type_feed
        self.parent_action_feed = parent_action_feed
        # self.srng = RandomStreams()

        # random initialize embeddings as normal distribution
        self.rule_embedding_W = torch.randn(rule_num, rule_embed_dim) * 0.1
        self.rule_embedding_b = torch.zeros(rule_num)
        self.node_embedding = torch.randn(node_num, node_embed_dim)
        self.vocab_embedding_W = torch.randn(target_vocab_size, rule_embed_dim)
        self.vocab_embedding_b = torch.zeros(target_vocab_size)

        self.query_tokens = None
        self.query_token_embed = None
        self.query_token_embed_mask = None
        self.query_embed = None

        # encoder output
        self.tgt_prob = None
        # decoder output
        self.completed_hyps = None

        # generate softmax output
        self.fc = nn.Linear(decoder_hidden_dim, output_dim)

        # TODO: what is PointerNet()?
        # might be: score calculator
        self.src_ptr_net = PointerNet()

    def forward(self, inputs):
        raise NotImplemented

    def encode(self, query_tokens, tgt_action_seq, tgt_action_seq_type, tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq):
        self.query_tokens = query_tokens

        query_tokens = torch.LongTensor(query_tokens)
        tgt_action_seq = torch.LongTensor(tgt_action_seq)
        tgt_action_seq_type = torch.LongTensor(tgt_action_seq_type)
        tgt_node_seq = torch.LongTensor(tgt_node_seq)
        tgt_par_rule_seq = torch.LongTensor(tgt_par_rule_seq)
        tgt_par_t_seq = torch.LongTensor(tgt_par_t_seq)

        # (batch_size, max_example_action_num, symbol_embed_dim)
        tgt_node_embed = torch.stack([self.node_embedding[tgt_node_batch] for tgt_node_batch in tgt_node_seq.tolist()])

        self.init_query(query_tokens)

        # previous action embeddings
        # (batch_size, max_example_action_num, action_embed_dim)
        tgt_action_seq_embed = torch.where(tgt_action_seq[:, :, 0].unsqueeze(2) > 0,
                                           torch.stack([self.rule_embedding_W[action[:, 0]] for action in tgt_action_seq]),
                                           torch.stack([self.vocab_embedding_W[action[:, 1]] for action in tgt_action_seq]))

        tgt_action_seq_embed_tm1 = tensor_right_shift(tgt_action_seq_embed)

        # parent rule application embeddings
        tgt_par_rule_embed = torch.where(tgt_par_rule_seq[:, :, None] < 0,
                                         torch.zeros(self.rule_embed_dim),
                                         # T.alloc(0., 1, config.rule_embed_dim),
                                         torch.stack([self.rule_embedding_W[rule] for rule in tgt_par_rule_seq]))

        if not self.frontier_node_type_feed:
            tgt_node_embed *= 0.

        if not self.parent_action_feed:
            tgt_par_rule_embed *= 0.

        # (batch_size, max_example_action_num, action_embed_dim + symbol_embed_dim + action_embed_dim)
        decoder_input = torch.cat((tgt_action_seq_embed_tm1, tgt_node_embed, tgt_par_rule_embed), dim=-1)

        # (batch_size, max_example_action_num)
        tgt_action_seq_mask = torch.any(tgt_action_seq_type.to(dtype=torch.uint8), dim=-1)

        decoder_hidden_state_trans_rule, decoder_hidden_state_trans_token = self.decoder(
            [decoder_input, self.query_embed, self.query_token_embed_mask, tgt_action_seq_mask, tgt_par_t_seq])

        # (batch_size, max_example_action_num, rule_num)
        # TODO: use dot or matmul?
        rule_predict = F.softmax(torch.matmul(decoder_hidden_state_trans_rule,
                                               torch.t(self.rule_embedding_W)) + self.rule_embedding_b)

        # (batch_size, max_example_action_num, 2)
        terminal_gen_action_prob = F.softmax(self.fc(self.decoder.hidden))

        # (batch_size, max_example_action_num, target_vocab_size)
        # TODO: use dot or matmul?
        vocab_predict = F.softmax(torch.matmul(decoder_hidden_state_trans_token, torch.t(
            self.vocab_embedding_W)) + self.vocab_embedding_b)

        # (batch_size, max_example_action_num, lstm_hidden_state + encoder_hidden_dim)
        ptr_net_decoder_state = torch.cat((self.decoder.hidden, self.decoder.ctx_vectors), dim=-1)

        # (batch_size, max_example_action_num, max_query_length)
        copy_prob = self.src_ptr_net(self.query_embed, self.query_token_embed_mask, ptr_net_decoder_state)

        # (batch_size, max_example_action_num)
        rule_tgt_prob = rule_predict[:, :, tgt_action_seq[:, :, 0]]

        # (batch_size, max_example_action_num)
        vocab_tgt_prob = vocab_predict[:, :, tgt_action_seq[:, :, 1]]

        # (batch_size, max_example_action_num)
        copy_tgt_prob = copy_prob[:, :, tgt_action_seq[:, :, 2]]

        # (batch_size, max_example_action_num)
        tgt_action_seq_type = tgt_action_seq_type.float()
        self.tgt_prob = tgt_action_seq_type[:, :, 0] * rule_tgt_prob + \
                        tgt_action_seq_type[:, :, 1] * terminal_gen_action_prob[:, :, 0] * vocab_tgt_prob + \
                        tgt_action_seq_type[:, :, 2] * terminal_gen_action_prob[:, :, 1] * copy_tgt_prob

        tgt_action_seq_mask = tgt_action_seq_mask.float()
        likelihood = torch.log(self.tgt_prob + 1.e-7 * (1. - tgt_action_seq_mask))
        # / tgt_action_seq_mask.sum(axis=-1)
        loss = - (likelihood * tgt_action_seq_mask).sum(dim=-1)
        loss = loss.mean()

        # if loss == float('inf'):
        #     print("="*60)
        #     print(self.tgt_prob)
        #     print("=" * 60)
        #     pdb.set_trace()

        return loss

    def init_query(self, query_tokens):
        # (batch_size, max_query_length, query_token_embed_dim)
        # (batch_size, max_query_length)
        self.query_token_embed = self.encoder.embedding(query_tokens)
        self.query_token_embed_mask = get_mask(query_tokens)
        self.query_embed = self.encoder(self.query_token_embed, mask=self.query_token_embed_mask)
        # self.query_embed = self.encoder(query_tokens, self.query_token_embed,
        #                                 mask=self.query_token_embed_mask, dropout=self.dropout, srng=self.srng)
        return self.query_embed, self.query_token_embed_mask

    def next_step(self, time_steps, decoder_prev_state, decoder_prev_cell, hist_h, prev_action_embed, node_id,
                  par_rule_id, parent_t, query_embed, query_token_embed_mask):
        # (batch_size, node_embed_dim)
        node_embed = self.node_embedding[node_id]
        # (batch_size, decoder_state_dim)
        par_rule_embed = torch.where(par_rule_id[:, None] < 0,
                                     torch.zeros(self.rule_embed_dim),
                                     self.rule_embedding_W[par_rule_id])
        decoder_next_state, decoder_next_cell, decoder_next_state_trans_rule, decoder_next_state_trans_token = self.decoder.next_step(
            [node_embed, par_rule_embed, prev_action_embed, time_steps, parent_t, decoder_prev_state, decoder_prev_cell,
             hist_h, query_embed, query_token_embed_mask])
        rule_prob = F.softmax(torch.matmul(decoder_next_state_trans_rule, torch.t(
            self.rule_embedding_W)) + self.rule_embedding_b)

        gen_action_prob = self.fc(decoder_next_state)

        vocab_prob = F.softmax(torch.matmul(decoder_next_state_trans_token, torch.t(
            self.vocab_embedding_W)) + self.vocab_embedding_b)

        ptr_net_decoder_state = torch.cat([decoder_next_state, self.decoder.ctx_vectors], dim=-1)

        copy_prob = self.src_ptr_net(query_embed, query_token_embed_mask, ptr_net_decoder_state)

        copy_prob = copy_prob.reshape(copy_prob.shape[0], -1)

        return decoder_next_state, decoder_next_cell, rule_prob, gen_action_prob, vocab_prob, copy_prob

    def decode(self, example, grammar, terminal_vocab, beam_size, max_time_step, log=False):
        # beam search decoding
        eos = 1
        unk = terminal_vocab.unk
        vocab_embedding = self.vocab_embedding_W
        rule_embedding = self.rule_embedding_W

        query_tokens = torch.LongTensor(example.data[0])

        self.init_query(query_tokens)

        completed_hyps = []
        completed_hyp_num = 0
        live_hyp_num = 1

        root_hyp = Hyp(grammar)
        root_hyp.state = torch.zeros(self.decoder.hidden_dim)
        root_hyp.cell = torch.zeros(self.decoder.hidden_dim)
        root_hyp.action_embed = torch.zeros(self.rule_embed_dim)
        root_hyp.node_id = grammar.get_node_type_id(root_hyp.tree.type)
        root_hyp.parent_rule_id = -1

        hyp_samples = [root_hyp]  # [list() for i in range(live_hyp_num)]

        # source word id in the terminal vocab
        src_token_id = [terminal_vocab[t] for t in example.query][:self.max_query_length]
        unk_pos_list = [x for x, t in enumerate(src_token_id) if t == unk]

        # sometimes a word may appear multi-times in the source, in this case,
        # we just copy its first appearing position. Therefore we mask the words
        # appearing second and onwards to -1
        token_set = set()
        for i, tid in enumerate(src_token_id):
            if tid in token_set:
                src_token_id[i] = -1
            else:
                token_set.add(tid)

        for t in range(max_time_step):
            hyp_num = len(hyp_samples)
            # print 'time step [%d]' % t
            decoder_prev_state = torch.stack([hyp.state for hyp in hyp_samples])
            decoder_prev_cell = torch.stack([hyp.cell for hyp in hyp_samples])

            hist_h = torch.zeros((hyp_num, max_time_step, self.decoder.hidden_dim))

            if t > 0:
                for i, hyp in enumerate(hyp_samples):
                    hist_h[i, :len(hyp.hist_h), :] = hyp.hist_h
                    # for j, h in enumerate(hyp.hist_h):
                    #    hist_h[i, j] = h

            prev_action_embed = torch.stack([hyp.action_embed for hyp in hyp_samples])
            node_id = torch.LongTensor([hyp.node_id for hyp in hyp_samples])
            parent_rule_id = torch.LongTensor([hyp.parent_rule_id for hyp in hyp_samples])
            parent_t = torch.LongTensor([hyp.get_action_parent_t() for hyp in hyp_samples]).unsqueeze(0)
            query_embed_tiled = self.query_embed.repeat(live_hyp_num, 1, 1)
            query_token_embed_mask_tiled = self.query_token_embed_mask.repeat(live_hyp_num, 1)

            inputs = [t, decoder_prev_state, decoder_prev_cell, hist_h, prev_action_embed,
                      node_id, parent_rule_id, parent_t,
                      query_embed_tiled, query_token_embed_mask_tiled]

            decoder_next_state, decoder_next_cell, rule_prob, gen_action_prob, vocab_prob, copy_prob = \
                self.next_step(*inputs)

            new_hyp_samples = []

            cut_off_k = beam_size
            score_heap = []

            # iterating over items in the beam
            # print 'time step: %d, hyp num: %d' % (t, live_hyp_num)

            word_prob = gen_action_prob[:, 0:1] * vocab_prob
            word_prob[:, unk] = 0

            hyp_scores = torch.Tensor([hyp.score for hyp in hyp_samples])

            # word_prob[:, src_token_id] += gen_action_prob[:, 1:2] * copy_prob[:, :len(src_token_id)]
            # word_prob[:, unk] = 0

            rule_apply_cand_hyp_ids = []
            rule_apply_cand_scores = []
            rule_apply_cand_rules = []
            rule_apply_cand_rule_ids = []

            hyp_frontier_nts = []
            word_gen_hyp_ids = []
            cand_copy_probs = []
            unk_words = []

            for k in range(live_hyp_num):
                hyp = hyp_samples[k]

                # if k == 0:
                #     print 'Top Hyp: %s' % hyp.tree.__repr__()

                frontier_nt = hyp.frontier_nt()
                hyp_frontier_nts.append(frontier_nt)

                assert hyp, 'none hyp!'

                # if it's not a leaf
                if not grammar.is_value_node(frontier_nt):
                    # iterate over all the possible rules
                    rules = grammar[frontier_nt.as_type_node] if self.head_nt_constraint else grammar
                    assert len(rules) > 0, 'fail to expand nt node %s' % frontier_nt
                    for rule in rules:
                        rule_id = grammar.rule_to_id[rule]

                        cur_rule_score = torch.log(rule_prob[k, rule_id])
                        new_hyp_score = hyp.score + cur_rule_score

                        rule_apply_cand_hyp_ids.append(k)
                        rule_apply_cand_scores.append(new_hyp_score)
                        rule_apply_cand_rules.append(rule)
                        rule_apply_cand_rule_ids.append(rule_id)

                else:  # it's a leaf that holds values
                    cand_copy_prob = 0.0
                    for i, tid in enumerate(src_token_id):
                        if tid != -1:
                            word_prob[k, tid] += gen_action_prob[k, 1] * copy_prob[k, i]
                            cand_copy_prob = gen_action_prob[k, 1]

                    # and unk copy probability
                    if len(unk_pos_list) > 0:
                        unk_pos = copy_prob[k, unk_pos_list].argmax()
                        unk_pos = unk_pos_list[unk_pos]

                        unk_copy_score = gen_action_prob[k, 1] * copy_prob[k, unk_pos]
                        word_prob[k, unk] = unk_copy_score

                        unk_word = example.query[unk_pos]
                        unk_words.append(unk_word)

                        cand_copy_prob = gen_action_prob[k, 1]

                    word_gen_hyp_ids.append(k)
                    cand_copy_probs.append(cand_copy_prob)

            # prune the hyp space
            if completed_hyp_num >= beam_size:
                break

            word_prob = torch.log(word_prob)

            word_gen_hyp_num = len(word_gen_hyp_ids)
            rule_apply_cand_num = len(rule_apply_cand_scores)

            if word_gen_hyp_num > 0:
                word_gen_cand_scores = hyp_scores[word_gen_hyp_ids,
                                                  None] + word_prob[word_gen_hyp_ids, :]
                word_gen_cand_scores_flat = word_gen_cand_scores.flatten()

                cand_scores = np.concatenate([rule_apply_cand_scores, word_gen_cand_scores_flat])
            else:
                cand_scores = np.array(rule_apply_cand_scores)

            top_cand_ids = (-cand_scores).argsort()[:beam_size - completed_hyp_num]

            # expand_cand_num = 0
            for cand_id in top_cand_ids:
                # cand is rule application
                new_hyp = None
                if cand_id < rule_apply_cand_num:
                    hyp_id = rule_apply_cand_hyp_ids[cand_id]
                    hyp = hyp_samples[hyp_id]
                    rule_id = rule_apply_cand_rule_ids[cand_id]
                    rule = rule_apply_cand_rules[cand_id]
                    new_hyp_score = rule_apply_cand_scores[cand_id]

                    new_hyp = Hyp(hyp)
                    new_hyp.apply_rule(rule)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = rule_embedding[rule_id]
                else:
                    tid = (cand_id - rule_apply_cand_num) % word_prob.shape[1]
                    word_gen_hyp_id = (cand_id - rule_apply_cand_num) / word_prob.shape[1]
                    hyp_id = word_gen_hyp_ids[word_gen_hyp_id]

                    if tid == unk:
                        token = unk_words[word_gen_hyp_id]
                    else:
                        token = terminal_vocab.id_token_map[tid]

                    frontier_nt = hyp_frontier_nts[hyp_id]
                    # if frontier_nt.type == int and (not (is_numeric(token) or token == '<eos>')):
                    #     continue

                    hyp = hyp_samples[hyp_id]
                    new_hyp_score = word_gen_cand_scores[word_gen_hyp_id, tid]

                    new_hyp = Hyp(hyp)
                    new_hyp.append_token(token)

                    if log:
                        cand_copy_prob = cand_copy_probs[word_gen_hyp_id]
                        if cand_copy_prob > 0.5:
                            new_hyp.log += ' || ' + \
                                str(new_hyp.frontier_nt()) + \
                                '{copy[%s][p=%f]}' % (token, cand_copy_prob)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = vocab_embedding[tid]
                    new_hyp.node_id = grammar.get_node_type_id(frontier_nt)

                # get the new frontier nt after rule application
                new_frontier_nt = new_hyp.frontier_nt()

                # if new_frontier_nt is None, then we have a new completed hyp!
                if new_frontier_nt is None:
                    # if t <= 1:
                    #     continue

                    new_hyp.n_timestep = t + 1
                    completed_hyps.append(new_hyp)
                    completed_hyp_num += 1

                else:
                    new_hyp.node_id = grammar.get_node_type_id(new_frontier_nt.type)
                    # new_hyp.parent_rule_id = grammar.rule_to_id[
                    #     new_frontier_nt.parent.to_rule(include_value=False)]
                    new_hyp.parent_rule_id = grammar.rule_to_id[new_frontier_nt.parent.applied_rule]

                    new_hyp_samples.append(new_hyp)

                # expand_cand_num += 1
                # if expand_cand_num >= beam_size - completed_hyp_num:
                #     break

                # cand is word generation

            live_hyp_num = min(len(new_hyp_samples), beam_size - completed_hyp_num)
            if live_hyp_num < 1:
                break

            hyp_samples = new_hyp_samples
            # hyp_samples = sorted(new_hyp_samples, key=lambda x: x.score, reverse=True)[:live_hyp_num]

        self.completed_hyps = sorted(completed_hyps, key=lambda x: x.score, reverse=True)

        return self.completed_hyps

    @property
    def params_name_to_id(self):
        name_to_id = dict()
        for i, p in enumerate(self.params):
            assert p.name is not None
            # print 'parameter [%s]' % p.name

            name_to_id[p.name] = i

        return name_to_id

    @property
    def params_dict(self):
        assert len(set(p.name for p in self.params)) == len(self.params), 'param name clashes!'
        return OrderedDict((p.name, p) for p in self.params)

    def pull_params(self):
        return OrderedDict([(p_name, p) for (p_name, p) in self.params_dict.iteritems()])

    def save(self, model_file, **kwargs):
        logging.info('save model to [%s]', model_file)

        weights_dict = self.pull_params()
        for k, v in kwargs.iteritems():
            weights_dict[k] = v

        np.savez(model_file, **weights_dict)

    def load(self, model_file):
        logging.info('load model from [%s]', model_file)
        weights_dict = np.load(model_file)

        # assert len(weights_dict.files) == len(self.params_dict)

        for p_name, p in self.params_dict.iteritems():
            if p_name not in weights_dict:
                raise RuntimeError('parameter [%s] not in saved weights file', p_name)
            else:
                logging.info('loading parameter [%s]', p_name)
                assert np.array_equal(p.shape.eval(), weights_dict[p_name].shape), \
                    'shape mis-match for [%s]!, %s != %s' % (p_name,
                                                             p.shape.eval(), weights_dict[p_name].shape)

                p.set_value(weights_dict[p_name])

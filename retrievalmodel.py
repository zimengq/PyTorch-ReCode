import numpy as np
import logging
import copy

from seq2seq.seq2seq import Seq2Seq

from config import config_info
import config
from lang.grammar import Grammar
from parse import *
from astnode import *
from components import Hyp, PointerNet, CondAttLSTM
from retrieval import NGramSearcher

import torch


class CondAttLSTMAligner(CondAttLSTM):

    def __init__(self, *args, **kwargs):
        super(CondAttLSTMAligner, self).__init__(*args, **kwargs)

    def activation(self, Type):
        activation = {'tah'}

    def _attention_over_history(self, hist_h, h_tm1, scores):
        hist_h_mask = torch.ones(hist_h.size())

        hist_h_att_trans = torch.matmul(hist_h, self.hatt_hist_W1) + self.hatt_b1
        h_tm1_hatt_trans = torch.matmul(h_tm1, self.hatt_h_W1)

        hatt_hidden = hist_h_att_trans + h_tm1_hatt_trans[:, None, :]
        hatt_hidden = hatt_hidden.tanh()
        hatt_raw = (torch.matmul(hatt_hidden, self.hatt_W2) + self.hatt_b2).view(hist_h.size()[0], hist_h.size()[1])
        hatt_exp = torch.exp(hatt_raw - torch.mul(torch.max(hatt_raw, dim=-1, keepdim=True)), hist_h_mask)
        h_att_weights = torch.div(hatt_exp, torch.sum(hatt_exp, dim=-1, keepdim=True) + 1e-7)

        # batch, output_dim
        ctx = torch.sum(torch.mul(hist_h, scores), dim=-1)
        return ctx

    def _step_align(self,
                    t, xi_t, xf_t, xo_t, xc_t, mask_t, parent_t,
                    h_tm1, c_tm1, hist_h,
                    u_i, u_f, u_o, u_c,
                    c_i, c_f, c_o, c_c,
                    h_i, h_f, h_o, h_c,
                    p_i, p_f, p_o, p_c,
                    att_h_w1, att_w2, att_b2,
                    context, context_mask, context_att_trans,
                    b_u):

        # batch, attn_lyaer1_dim
        h_tm1_att_trans = torch.matmul(h_tm1, att_h_w1)

        att_hidden = context_att_trans+ h_tm1_att_trans[:, None, :]
        att_hidden = att_hidden.tanh()

        att_raw = torch.matmul(att_hidden, att_w2) + att_b2
        # TODO necessary?
        att_raw = att_raw.view(att_raw.size()[0], att_raw.size()[1])

        # batch, context_size
        ctx_att = torch.exp(att_raw - torch.max(att_raw, dim=-1, keepdims=True)[0])

        if context_mask:
            ctx_att = torch.mul(ctx_att, context_mask)

        # TODO different from source code
        ctx_att = torch.div(ctx_att, torch.sum(ctx_att, dim=-1, keepdim=True))

        # batch, context_dim
        scores = ctx_att[:, :, None]
        ctx_vec = torch.sum(torch.mul(context, scores), dim=-1)

        if t:
            h_ctx_vec = self._attention_over_history(hist_h, h_tm1, scores)
        else:
            h_ctx_vec = torch.zeros(h_tm1.size())

        if not config.parent_hidden_state_feed:
            t = 0

        if t:
            par_h = hist_h[:hist_h.size()[0], parent_t, :]
        else:
            par_h = torch.zeros(h_tm1.size())

        if config.tree_attention:
            i_t = self.inner_activation(xi_t + torch.matmul(h_tm1 * b_u[0], u_i) + torch.matmul(ctx_vec, c_i) + torch.matmul(par_h, p_i) + torch.matmul(h_ctx_vec, h_i))
            f_t = self.inner_activation(xf_t + torch.matmul(h_tm1 * b_u[1], u_f) + torch.matmul(ctx_vec, c_f) + torch.matmul(par_h, p_f) + torch.matmul(h_ctx_vec, h_f))
            c_t = f_t * c_tm1 + i_t * self.activation(xc_t + torch.matmul(h_tm1 * b_u[2], u_c) + torch.matmul(ctx_vec, c_c) + torch.matmul(par_h, p_c) + torch.matmul(h_ctx_vec, h_c))
            o_t = self.inner_activation(xo_t + torch.matmul(h_tm1 * b_u[3], u_o) + torch.matmul(ctx_vec, c_o) + torch.matmul(par_h, p_o) + torch.matmul(h_ctx_vec, h_o))
        else:
            i_t = self.inner_activation(xi_t + torch.matmul(h_tm1 * b_u[0], u_i) + torch.matmul(ctx_vec, c_i) + torch.matmul(par_h, p_i))
            f_t = self.inner_activation(xf_t + torch.matmul(h_tm1 * b_u[1], u_f) + torch.matmul(ctx_vec, c_f) + torch.matmul(par_h, p_f))
            c_t = f_t * c_tm1 + i_t * self.activation(xc_t + torch.matmul(h_tm1 * b_u[2], u_c) + torch.matmul(ctx_vec, c_c) + torch.matmul(par_h, p_c))
            o_t = self.inner_activation(xo_t + torch.matmul(h_tm1 * b_u[3], u_o) + torch.matmul(ctx_vec, c_o) + torch.matmul(par_h, p_o))
        h_t = o_t * self.activation(c_t)

        h_t = (1 - mask_t) * h_tm1 + mask_t * h_t
        c_t = (1 - mask_t) * c_tm1 + mask_t * c_t

        hist_h[:, t, :] = h_t
        new_hist_h = hist_h[:, t, :]

        return h_t, c_t, scores, new_hist_h

    def align(self, x, context, parent_t_seq, init_state=None, init_cell=None, hist_h=None,
              mask=None, context_mask=None, srng=None, time_steps=None):
        assert context_mask.dtype == 'int8', 'context_mask is not int8, got %s' % context_mask.dtype

        x = x.permute((1, 0, 2))

        B_w = torch.ones((4,))
        B_u = torch.ones((4,))

        # (batch_size, context_size, att_layer1_dim)
        context_att_trans = torch.matmul(context, self.att_ctx_W1) + self.att_b1

        if not hist_h:
            # (batch_size, n_timestep, output_dim)
            hist_h = torch.zeros(x.size()[1], x.size()[0], self.output_dim)


        n_timestep = x.size()[0]
        time_steps = list(np.range(n_timestep))

        # (n_timestep, batch_size)
        parent_t_seq = parent_t_seq.permute((1, 0))

        for i in time_steps:
            outputs, cells, att_scores, hist_h_outputs = \
                self._step_align(self.U_i, self.U_f, self.U_o, self.U_c,
                                 self.C_i, self.C_f, self.C_o, self.C_c,
                                 self.H_i, self.H_f, self.H_o, self.H_c,
                                 self.P_i, self.P_f, self.P_o, self.P_c,
                                 self.att_h_W1, self.att_W2, self.att_b2,
                                 context, context_mask, context_att_trans,
                                 B_u)

        att_scores = att_scores.permute(1, 0, 2)

        return att_scores


class RetrievalModel(Seq2Seq):
    def __init__(self, encoder, decoder, decoder_hidden_dim, rule_num, rule_embed_dim,
                 node_num, node_embed_dim, target_vocab_size, max_query_length, head_nt_constraint,
                 output_dim=2, dropout=0.2, frontier_node_type_feed=False, parent_action_feed=False,
                 max_regular_model=None):
        """
        super(RetrievalModel, self).__init__()
        self.decoder_lstm = CondAttLSTMAligner(config.rule_embed_dim + config.node_embed_dim + config.rule_embed_dim,
                                               config.decoder_hidden_dim, config.encoder_hidden_dim, config.attention_hidden_dim,
                                               name='decoder_lstm')
        """
        super(RetrievalModel, self).__init__(encoder, decoder, decoder_hidden_dim, rule_num, rule_embed_dim,
                 node_num, node_embed_dim, target_vocab_size, max_query_length, head_nt_constraint,
                 output_dim=2, dropout=0.2, frontier_node_type_feed=False, parent_action_feed=False)

        self.decoder_lstm = CondAttLSTMAligner(config.rule_embed_dim + config.node_embed_dim + config.rule_embed_dim,
                                               config.decoder_hidden_dim, config.encoder_hidden_dim, config.attention_hidden_dim,
                                               name='decoder_lstm')
        # update params for new decoder
        # self.params = self.query_embedding.params + self.query_encoder_lstm.params + \
        #     self.decoder_lstm.params + self.src_ptr_net.params + self.terminal_gen_softmax.params + \
        #     [self.rule_embedding_W, self.rule_embedding_b, self.node_embedding, self.vocab_embedding_W, self.vocab_embedding_b] + \
        #     self.decoder_hidden_state_W_rule.params + self.decoder_hidden_state_W_token.params

    def build(self):
        super(RetrievalModel, self).build()
        self.build_aligner()

    def build_aligner(self):
        tgt_action_seq = torch.zeros(1, 1, 1)
        tgt_action_seq_type = torch.zeros(1, 1, 1)
        tgt_node_seq = torch.zeros(1, 1)
        tgt_par_rule_seq = torch.zeros(1, 1)
        tgt_par_t_seq = torch.zeros(1, 1, 1)

        tgt_node_embed = self.node_embedding[tgt_node_seq]

        # TODO what is query_token
        query_tokens = self.query_token
        query_token_embed, query_token_embed_mask = self.query_embedding(
            query_tokens, mask_zero=True)
        batch_size = tgt_action_seq.size()[0]
        max_example_action_num = tgt_action_seq.size()[1]

        tgt_action_pos = tgt_action_seq[:, :, 0] > 0
        tgt_action_pos_ = torch.cat((torch.ones(tgt_action_pos.size()[0]), tgt_action_pos.float()), dim=0)
        if tgt_action_pos_:
            tgt_action_seq_embed = self.rule_embedding_W[tgt_action_seq[:, :, 0]]
        else:
            tgt_action_seq_embed = self.vocab_embedding_W[tgt_action_seq[:, :, 1]]


        tgt_action_seq_embed_tm1 =  torch.zeros(tgt_action_seq_embed.size())
        tgt_action_seq_embed_tm1[:, 1:, :] = tgt_action_seq_embed[:, :-1, :]

        if tgt_par_rule_seq[:, :, None] < 0:
            tgt_par_rule_embed = torch.zeros(1, config.rule_embed_dim)
        else:
            tgt_par_rule_embed = self.rule_embedding_W[tgt_par_rule_seq]

        if not config.frontier_node_type_feed:
            tgt_node_embed *= 0.
        if not config.parent_action_feed:
            tgt_par_rule_embed *= 0.

        self.tgt_action_seq_embed_tm1 = tgt_action_seq_embed_tm1
        self.tgt_node_embed = tgt_node_embed
        self.tgt_par_rule_embed = tgt_par_rule_embed
        self.query_token_embed = query_token_embed
        self.query_token_embed_mask = query_token_embed_mask

    def align(self, alignment_inputs):
        query_tokens, tgt_action_seq, tgt_action_seq_type, tgt_node_seq, tgt_par_rule_seq, tgt_par_t_seq = alignment_inputs
        decoder_input = torch.cat((self.tgt_action_seq_embed_tm1, self.tgt_node_embed, self.tgt_par_rule_embed), dim=-1)
        query_embed = self.query_encoder_lstm(self.query_token_embed, mask=self.query_token_embed_mask, dropout=0, srng=self.srng)
        tgt_action_seq_mask = torch.any(tgt_action_seq_type, axis=-1)
        return self.decoder_lstm.align(decoder_input, context=query_embed,\
                context_mask=self.query_token_embed_mask,\
                mask=tgt_action_seq_mask,\
                parent_t_seq=tgt_par_t_seq,\
                srng=self.srng)

    def decode_with_retrieval(self, example, grammar, terminal_vocab, ngram_searcher, beam_size, max_time_step,
                              log=False):
        # beam search decoding with ngram retrieval
        eos = terminal_vocab.eos
        unk = terminal_vocab.unk
        vocab_embedding = self.vocab_embedding_W
        rule_embedding = self.rule_embedding_W

        query_tokens = example.data[0]
        self.init_query(torch.from_numpy(query_tokens).long())
        completed_hyps = []
        completed_hyp_num = 0
        live_hyp_num = 1

        root_hyp = Hyp_ng(grammar)
        root_hyp.state = torch.zeros(self.decoder.hidden_dim)
        root_hyp.cell = torch.zeros(self.decoder.hidden_dim)
        root_hyp.action_embed = torch.zeros(self.rule_embed_dim)
        root_hyp.node_id = grammar.get_node_type_id(root_hyp.tree.type)
        root_hyp.parent_rule_id = -1

        hyp_samples = [root_hyp]
        src_token_id = [terminal_vocab[t] for t in example.query][:config.max_query_length]
        unk_pos_list = [x for x, t in enumerate(src_token_id) if t == unk]

        token_set = set()
        for i, tid in enumerate(src_token_id):
            if tid in token_set:
                src_token_id[i] = -1
            else:
                token_set.add(tid)

        for t in xrange(max_time_step):
            hyp_num = len(hyp_samples)
            decoder_prev_state = torch.stack([hyp.state for hyp in hyp_samples])
            decoder_prev_cell = torch.stack([hyp.cell for hyp in hyp_samples])
            hist_h = torch.zeros((hyp_num, max_time_step, self.decoder.hidden_dim))

            if t > 0:
                for i, hyp in enumerate(hyp_samples):
                    hist_h[i, :len(hyp.hist_h), :] = torch.stack(hyp.hist_h)


            prev_action_embed = torch.stack([torch.tensor(hyp.action_embed, dtype=torch.long) for hyp in hyp_samples])
            node_id = torch.LongTensor([hyp.node_id for hyp in hyp_samples])
            parent_rule_id = torch.LongTensor([hyp.parent_rule_id for hyp in hyp_samples])
            parent_t = torch.LongTensor([hyp.get_action_parent_t() for hyp in hyp_samples]).unsqueeze(0)
            query_embed_tiled = self.query_embed.repeat(live_hyp_num, 1, 1)
            query_token_embed_mask_tiled = self.query_token_embed_mask.repeat(live_hyp_num, 1)

            inputs = [t, decoder_prev_state, decoder_prev_cell, hist_h, prev_action_embed,
                      node_id, parent_rule_id, parent_t,
                      query_embed_tiled, query_token_embed_mask_tiled]

            decoder_next_state, decoder_next_cell, rule_prob, gen_action_prob, vocab_prob, copy_prob = self.next_step(*inputs)
            rule_prob, vocab_prob, copy_prob = update_probs(rule_prob, vocab_prob, copy_prob, hyp_samples, ngram_searcher, grammar=grammar)

            new_hyp_samples = []
            word_prob = gen_action_prob[:, 0:1] * vocab_prob
            word_prob[:, unk] = 0

            hyp_scores = torch.Tensor([hyp.score for hyp in hyp_samples])

            rule_apply_cand_hyp_ids = []
            rule_apply_cand_scores = []
            rule_apply_cand_rules = []
            rule_apply_cand_rule_ids = []

            hyp_frontier_nts = []
            word_gen_hyp_ids = []
            cand_copy_probs = []
            unk_words = []

            for k in xrange(live_hyp_num):
                hyp = hyp_samples[k]

                frontier_nt = hyp.frontier_nt()
                hyp_frontier_nts.append(frontier_nt)

                if not grammar.is_value_node(frontier_nt):
                    rules = grammar[frontier_nt.as_type_node] if config.head_nt_constraint else grammar
                    assert len(rules) > 0, 'fail to expand nt node %s' % frontier_nt
                    for rule in rules:
                        rule_id = grammar.rule_to_id[rule]

                        cur_rule_score = torch.log(rule_prob[k, rule_id])
                        new_hyp_score = hyp.score + cur_rule_score

                        rule_apply_cand_hyp_ids.append(k)
                        rule_apply_cand_scores.append(new_hyp_score)
                        rule_apply_cand_rules.append(rule)
                        rule_apply_cand_rule_ids.append(rule_id)

                else:
                    cand_copy_prob = 0.0
                    for i, tid in enumerate(src_token_id):
                        if tid != -1:
                            word_prob[k, tid] += gen_action_prob[k, 1] * copy_prob[k, i]
                            cand_copy_prob = gen_action_prob[k, 1]

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

            if completed_hyp_num >= beam_size:
                break

            word_prob = torch.log(word_prob)

            word_gen_hyp_num = len(word_gen_hyp_ids)
            rule_apply_cand_num = len(rule_apply_cand_scores)

            if word_gen_hyp_num > 0:
                word_gen_cand_scores = hyp_scores[word_gen_hyp_ids,
                                                  None] + word_prob[word_gen_hyp_ids, :]
                word_gen_cand_scores_flat = word_gen_cand_scores.flatten()

                cand_scores = torch.cat((torch.tensor(rule_apply_cand_scores), word_gen_cand_scores_flat))
            else:
                cand_scores = torch.tensor(rule_apply_cand_scores)

            top_cand_ids = (-cand_scores).argsort()[:beam_size - completed_hyp_num]

            for k, cand_id in enumerate(top_cand_ids):
                verbose = False
                if cand_id < rule_apply_cand_num:
                    hyp_id = rule_apply_cand_hyp_ids[cand_id]
                    hyp = hyp_samples[hyp_id]
                    rule_id = rule_apply_cand_rule_ids[cand_id]
                    rule = rule_apply_cand_rules[cand_id]
                    new_hyp_score = rule_apply_cand_scores[cand_id]

                    new_hyp = Hyp_ng(hyp)
                    new_hyp.apply_rule(rule)
                    new_hyp.to_move = False
                    new_hyp.update_ngrams(ngram_searcher.get_keys(
                        hyp.get_ngrams(), rule_id, "APPLY_RULE", verbose))
                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.hist_h.append(copy.copy(new_hyp.state))
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = rule_embedding[rule_id]
                else:
                    tid = (cand_id - rule_apply_cand_num) % word_prob.shape[1]
                    tid = tid.tolist()
                    word_gen_hyp_id = (cand_id - rule_apply_cand_num) / word_prob.shape[1]
                    hyp_id = word_gen_hyp_ids[word_gen_hyp_id]

                    if tid == unk:
                        token = unk_words[word_gen_hyp_id]
                    else:
                        token = terminal_vocab.id_token_map[tid]

                    frontier_nt = hyp_frontier_nts[hyp_id]
                    hyp = hyp_samples[hyp_id]
                    new_hyp_score = word_gen_cand_scores[word_gen_hyp_id, tid]
                    new_hyp = Hyp_ng(hyp)
                    new_hyp.append_token(token)

                    if tid == unk:
                        new_hyp.update_ngrams(ngram_searcher.get_keys(hyp.get_ngrams(),
                                                                      unk_pos, "COPY_TOKEN", verbose))
                    elif tid in src_token_id:
                        new_hyp.update_ngrams(ngram_searcher.get_keys(hyp.get_ngrams(),
                                                                      src_token_id.index(tid), "COPY_TOKEN",
                                                                      verbose))
                    else:
                        new_hyp.update_ngrams(ngram_searcher.get_keys(
                            hyp.get_ngrams(), tid, "GEN_TOKEN", verbose))

                    if tid == eos:
                        new_hyp.to_move = True
                    else:
                        new_hyp.to_move = False

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

                new_frontier_nt = new_hyp.frontier_nt()

                if new_frontier_nt is None:

                    new_hyp.n_timestep = t + 1
                    completed_hyps.append(new_hyp)
                    completed_hyp_num += 1

                else:
                    new_hyp.node_id = grammar.get_node_type_id(new_frontier_nt.type)
                    new_hyp.parent_rule_id = grammar.rule_to_id[new_frontier_nt.parent.applied_rule]
                    new_hyp_samples.append(new_hyp)

            live_hyp_num = min(len(new_hyp_samples), beam_size - completed_hyp_num)
            if live_hyp_num < 1:
                break

            hyp_samples = new_hyp_samples

        completed_hyps = sorted(completed_hyps, key=lambda x: x.score, reverse=True)

        return completed_hyps


class Hyp_ng(Hyp):
    def __init__(self, *args):
        super(Hyp_ng, self).__init__(*args)
        if isinstance(args[0], Hyp):
            self.hist_ng = copy.copy(args[0].hist_ng)
        else:
            self.hist_ng = []
        self.to_move = False

    def update_ngrams(self, new_ngram):
        # print new_ngram
        self.hist_ng.append(new_ngram)

    def get_ngrams(self, verbose=False):
        try:
            if self.to_move:
                t = self.get_action_parent_t()
            else:
                t = self.t

            k = self.hist_ng[t]
            if verbose:
                print self.to_move
                print t
                print len(self.hist_ng)
                print self.tree.pretty_print()
            return k
        except:
            return [None for _ in range(config.max_ngrams + 1)]


def update_probs(rule_prob, vocab_prob, copy_prob, hyp_samples, ngram_searcher, grammar=None):
    f = config.retrieval_factor

    for k, hyp in enumerate(hyp_samples):
        verbose = False
        ngram_keys = hyp.get_ngrams(verbose)
        for value, score, flag in ngram_searcher(ngram_keys):

            if flag == "APPLY_RULE":
                 rule_prob[k, value] = torch.mul(rule_prob[k, value], torch.exp(torch.tensor(f * score)))

            elif flag == "GEN_TOKEN":
                vocab_prob[k, value] = torch.mul(vocab_prob[k, value], torch.exp(torch.tensor(f * score)))

            else:
                assert flag == "COPY_TOKEN"
                if value < config.max_query_length:
                    copy_prob[k, value] = torch.mul(copy_prob[k, value], torch.exp(torch.tensor(f * score)))

        rule_prob[k] = torch.div(rule_prob[k], torch.sum(rule_prob[k]))
        vocab_prob[k] = torch.div(vocab_prob[k], torch.sum(vocab_prob[k]))
        copy_prob[k] = torch.div(copy_prob[k], torch.sum(copy_prob[k]))

    return rule_prob, vocab_prob, copy_prob

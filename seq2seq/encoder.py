import torch
import torch.nn as nn

from nn.layers.recurrent import *


class BaseEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, embedding=None):
        super(BaseEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)

    def forward(self, inputs):
        raise NotImplemented


class LSTMEncoder(BaseEncoder):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding=None, bidirectional=False, dropout=0.2):
        super(LSTMEncoder, self).__init__(vocab_size, embed_dim, embedding)
        self.bidirectional = bidirectional
        self.dropout = dropout

        if bidirectional:
            self.encoder = BiLSTM(embed_dim, hidden_dim / 2, return_sequences=True)
        else:
            self.encoder = LSTM(embed_dim, hidden_dim, return_sequences=True)

        self.query_embed = None

    def forward(self, query_token_embed, mask=None):
        # (batch_size, max_query_length, query_embed_dim)
        self.query_embed = self.encoder(query_token_embed, mask=mask, dropout=self.dropout)
        return self.query_embed


import torch
import torch.nn as nn


class SeqMyEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout, has_embedding=False, emb_dim=1):
        super().__init__()

        self.hid_dim = hid_dim
        self.hasEmbedding = has_embedding

        if self.hasEmbedding:
            self.embedding = nn.Embedding(input_dim, emb_dim)  # no dropout as only one layer!
            self.input_dim = emb_dim
        else:
            self.input_dim = input_dim

        self.rnn = nn.GRU(self.input_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [seq len, batch size]
        if self.hasEmbedding:
            embedded = self.dropout(self.embedding(src))  # embedded = [seq len, batch size, emb dim]
            input = embedded
        else:
            input = src

        outputs, hidden = self.rnn(input)  # no cell state!
        # outputs = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden

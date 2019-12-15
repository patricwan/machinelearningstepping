import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, dropout, has_embedding=False, emb_dim=1):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.hasEmbedding = has_embedding

        if self.hasEmbedding:
            self.embedding = nn.Embedding(output_dim, emb_dim)
            self.input_dim = emb_dim + hid_dim
        else:
            self.input_dim = hid_dim
        self.rnn = nn.GRU(self.input_dim, self.hid_dim)

        if self.hasEmbedding:
            self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        else:
            self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]
        if self.hasEmbedding:
            input = input.unsqueeze(0)
            # input = [1, batch size]
            embedded = self.dropout(self.embedding(input))
            # embedded = [1, batch size, emb dim]
            emb_con = torch.cat((embedded, context), dim=2)
            # emb_con = [1, batch size, emb dim + hid dim]
            input_data = emb_con
        else:
            input_data = input

        output, hidden = self.rnn(input_data, hidden)

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        if self.hasEmbedding:
            output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)

        # output = [batch size, emb dim + hid dim * 2]

        prediction = self.fc_out(output)

        # prediction = [batch size, output dim]

        return prediction, hidden


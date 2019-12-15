import torch
import torch.nn as nn
import random
import math
import time


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()


    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [src len, batch size]
        #trg = [trg len, batch size]

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, self.decoder.output_dim).to(self.device)

        #last hidden state of the encoder is the context
        context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, context)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
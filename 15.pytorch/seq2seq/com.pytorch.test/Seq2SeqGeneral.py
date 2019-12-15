import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, enc_input_dim, enc_hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(enc_input_dim, enc_hidden_dim)

    def forward(self, inputBatchData_i):
        # enc_ouput:  (seq_len, batch, enc_hidden_size)
        # enc_hid: (1, batch, enc_hidden_size)
        enc_output_i, enc_hid_i = self.rnn(inputBatchData_i)
        return enc_output_i, enc_hid_i

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, enc_output_i, enc_or_dec_hid_i):
        #print("attention started")
        enc_output_i = enc_output_i.permute(1, 0, 2)
        # enc_ouput:  (batch, seq_len, hidden_size)
        #print(enc_output_i.size())
        enc_or_dec_hid_i = enc_or_dec_hid_i.permute(1, 2, 0)
        # enc_hid: (batch, hidden_size, 1)
        #print(enc_or_dec_hid_i.size())

        firstMultiRes = enc_output_i.bmm(enc_or_dec_hid_i)
        # firstMultiRes (batch, seq_len, 1)
        #print(firstMultiRes.size())

        firstMultiRes = firstMultiRes.squeeze(2)
        #print(firstMultiRes.size())

        # weight (batch, seq_len)
        weight_i = F.softmax(firstMultiRes, dim=1)
        #print(weight_i.size())
        # print(weight_i[0,:])

        enc_output_i = enc_output_i.permute(0, 2, 1)
        # enc_ouput:  (batch, hidden_size, seq_len)
        #print("enc_output size", enc_output_i.size())

        weight_i = weight_i.unsqueeze(2)
        # weight (batch, seq_len, 1)
        #print("weight size ", weight_i.size())

        attentionRes = enc_output_i.bmm(weight_i)
        # weight (batch, hidden_size, 1)

        #print(attentionRes.size())
        attentionRes = attentionRes.squeeze(2)
        #weight(batch, hidden_size)

        #print("attentionRes size ", attentionRes.size())
        #print("attention stopped")
        return attentionRes

class Decoder(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, dec_target_temp_dim, dec_target_dim, attention):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_target_temp_dim = dec_target_temp_dim
        self.dec_target_dim = dec_target_dim

        self.rnn = nn.GRU(self.enc_hidden_dim + self.dec_target_dim, self.dec_hidden_dim)
        self.linear_out1 = nn.Linear(self.dec_hidden_dim, self.dec_target_temp_dim)
        self.linear_out2 = nn.Linear(self.dec_target_temp_dim, self.dec_target_dim)

        self.attention = attention

    def forward(self,oneInput_i, enc_output_i, dec_hid_i):
        #print("decoderFunctions starts")
        context = self.attention(enc_output_i, dec_hid_i)
        #print("context size", context.size())
        #print("oneInput size", oneInput_i.size())
        combinedInput = torch.cat((oneInput_i, context), dim=1)
        combinedInput = combinedInput.unsqueeze(1)
        combinedInput = combinedInput.permute(1, 0, 2)

        #print("combinedInput size ", combinedInput.size())
        #print("dec_hid_i size ", dec_hid_i.size())

        #print("GRU input dim ", self.enc_hidden_dim + self.dec_target_dim)
        #print("GRU hidden dim ", self.dec_hidden_dim)
        dec_output_i, dec_hid_i = self.rnn(combinedInput, dec_hid_i)

        #print("dec_output_i size ", dec_output_i.size())
        #print("dec_hid_i size ", dec_hid_i.size())

        target_output_temp = F.relu(self.linear_out1(dec_output_i))
        #print("target_output_temp size ", target_output_temp.size())

        target_output = F.relu(self.linear_out2(target_output_temp))
        #print("target_output size ", target_output.size())

        #print("decoderFunctions ends")
        return target_output, dec_hid_i

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, dec_seq_len, batch_size, dec_target_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.dec_seq_len = dec_seq_len
        self.batch_size = batch_size
        self.dec_target_dim = dec_target_dim

    def forward(self, batch_input_i, batch_output_i):
        #print("seq2seqFunctions starts")

        #print("batch_input_i size ", batch_input_i.size())
        #print("batch_output_i size ", batch_output_i.size())

        outputs_i = torch.zeros(self.dec_seq_len, self.batch_size, self.dec_target_dim)
        #print("outputs_i size ", outputs_i.size())

        encoder_outputs_i, temp_hidden_i = self.encoder(batch_input_i)
        #print("encoder_outputs_i size ", encoder_outputs_i.size())
        #print("encoder_hidden size ", temp_hidden_i.size())

        # get first input as target
        oneInput_i = batch_output_i[0, :, :]
        #print("oneInput_i size ", oneInput_i.size())

        for t in range(1, self.dec_seq_len):
            output_single_i, temp_hidden_i = self.decoder(oneInput_i, encoder_outputs_i, temp_hidden_i)
            outputs_i[t] = output_single_i
            oneInput_i = batch_output_i[t]
            #print("currently decoder time t ", t)

        #print("outputs_i size ", outputs_i.size())
        #print("seq2seqFunctions ends")
        return outputs_i

class Seq2SeqModelTrainer():
#  batch_size = 20, enc_input_dim = 8, enc_seq_len = 24, enc_hidden_dim = 12, enc_output_dim = 12,
# dec_input_dim = 2, dec_seq_len = 24, dec_hidden_dim = 12, dec_output_dim = 12,
#    dec_target_temp_dim = 6, dec_target_dim = 2, learning_rate = 0.001
    def __init__(self, iterator, paramsMap):
        self.batch_size = paramsMap.get("batch_size")

        self.enc_input_dim = paramsMap.get("enc_input_dim")
        self.enc_seq_len = paramsMap.get("enc_seq_len")
        self.enc_hidden_dim = paramsMap.get("enc_hidden_dim")
        self.enc_output_dim = paramsMap.get("enc_output_dim")

        self.dec_input_dim = paramsMap.get("dec_input_dim")
        self.dec_seq_len = paramsMap.get("dec_seq_len")
        self.dec_hidden_dim = paramsMap.get("dec_hidden_dim")
        self.dec_output_dim = paramsMap.get("dec_output_dim")

        self.dec_target_temp_dim = paramsMap.get("dec_target_temp_dim")
        self.dec_target_dim = paramsMap.get("dec_target_dim")

        self.learning_rate = paramsMap.get("learning_rate")

        self.encoder = Encoder(self.enc_input_dim,self.enc_hidden_dim)
        self.attention = Attention()
        self.decoder = Decoder(self.enc_hidden_dim, self.dec_hidden_dim,
                               self.dec_target_temp_dim, self.dec_target_dim, self.attention)

        self.seq2seqModel = Seq2Seq(self.encoder, self.decoder,
                                    self.dec_seq_len, self.batch_size, self.dec_target_dim)

        #Initialize Weights
        self.seq2seqModel.apply(self.init_weights)

        #self.optimzer = optim.Adam(self.seq2seqModel.parameters(), lr=learning_rate)
        self.optimzer = optim.SGD(self.seq2seqModel.parameters(), lr=paramsMap.get("learning_rate"))
        self.criterion = nn.MSELoss()

        self.epochsTotal = 100
        self.clip = 1
        self.iterator = iterator

    def init_weights(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean = 0, std = 0.01)
            else:
                nn.init.constant_(param.data, 0)

    def trainEach(self):
        self.seq2seqModel.train()
        epoch_loss = 0

        for i, batch in enumerate(self.iterator):
            batch_input = batch[0]
            batch_input = batch_input.permute(1,0,2)
            batch_target = batch[1]
            batch_target = batch_target.permute(1, 0, 2)

            self.optimzer.zero_grad()

            output = self.seq2seqModel(batch_input, batch_target)
            #[seq_len, batch_size, target_out_dim]

            output = output.reshape(-1,self.dec_target_dim)
            batch_target = batch_target.reshape(-1, self.dec_target_dim)
            # [batch_size * seq_len, target_out_dim]

            loss = self.criterion(output, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.seq2seqModel.parameters(), self.clip)

            self.optimzer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.iterator)

    def trainEpoch(self):
        for epoch in range(self.epochsTotal):
            train_loss = self.trainEach()

            print("epoch ", epoch)
            print("Train loss ", train_loss)

# BATCH_SIZE = 20
# ALL_RECORDS = 50000
#
# enc_input_dim = 8
# enc_seq_len = 24
# enc_hidden_dim = 12
# enc_output_dim = 12
#
# dec_input_dim = 2
# dec_seq_len = 24
# dec_hidden_dim = 12
# dec_output_dim = 12
#
# dec_target_temp_dim = 6
# dec_target_dim = 2

class AllParams():
    def __init__(self):
        self.paramsMap = {}
        self.paramsMap["batch_size"] = 20
        self.paramsMap["ALL_RECORDS"] = 50000
        self.paramsMap["learning_rate"] = 0.01

        self.paramsMap["enc_input_dim"] = 8
        self.paramsMap["enc_seq_len"] = 24
        self.paramsMap["enc_hidden_dim"] = 12
        self.paramsMap["enc_output_dim"] = 12

        self.paramsMap["dec_input_dim"] = 2
        self.paramsMap["dec_seq_len"] = 24
        self.paramsMap["dec_hidden_dim"] = 12
        self.paramsMap["dec_output_dim"] = 12

        self.paramsMap["dec_target_temp_dim"] = 6
        self.paramsMap["dec_target_dim"] = 2

    def get(self, paramName):
        return self.paramsMap[paramName]

import torch.utils.data as Data
import math
from torch.utils.data import Dataset

class DealDataset(Dataset):
    def __init__(self, allParams):
        self.src = torch.rand(allParams.get("enc_seq_len"), allParams.get("ALL_RECORDS"),
                              allParams.get("enc_input_dim")) + math.tanh(10)
        self.target = torch.rand(allParams.get("dec_seq_len"), allParams.get("ALL_RECORDS"),
                                allParams.get("dec_target_dim") ) + 4
        self.len = allParams.get("ALL_RECORDS")

    def __getitem__(self, index):
        return self.src[:,index,:], self.target[:,index,:]

    def __len__(self):
        return self.len

allParams = AllParams()

dealDataset = DealDataset(allParams)
training_data = Data.DataLoader(dataset=dealDataset, batch_size=allParams.get("batch_size"), shuffle=True)

seq2SeqModelTrainer = Seq2SeqModelTrainer(training_data, allParams)

for index, batch_data in enumerate(training_data):
    if (index % 1000 ==0):
        #print("index ", index)
        print("batch data src size ", batch_data[0].size())
        print("batch data target size ", batch_data[1].size())

seq2SeqModelTrainer.trainEpoch()

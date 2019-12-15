import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

batch_size = 20

enc_input_dim = 8
enc_seq_len = 24
enc_hidden_dim = 12
enc_output_dim = 12

dec_input_dim = 2
dec_seq_len = 24
dec_hidden_dim = 12
dec_output_dim = 12

dec_target_temp_dim = 6
dec_target_dim = 2

#encoder dimension Test
inputBatchData = torch.randn(enc_seq_len, batch_size, enc_input_dim)
print(inputBatchData.size())

outputBatchData = torch.randn(dec_seq_len, batch_size, dec_target_dim)
print(outputBatchData.size())

def encoderFunctions(inputBatchData_i):
    rnn = nn.GRU(enc_input_dim, enc_hidden_dim)
    #enc_ouput:  (seq_len, batch, num_directions * hidden_size)
    #enc_hid: (1, batch, hidden_size)
    enc_output_i, enc_hid_i = rnn(inputBatchData_i)
    return enc_output_i, enc_hid_i

enc_output, enc_hid = encoderFunctions(inputBatchData)
print(enc_output.size())
print(enc_hid.size())

#attention dimension Test
def attention(enc_output_i, dec_hid_i):
    print("attention started")
    enc_output_i = enc_output_i.permute(1,0,2)
    # enc_ouput:  (batch, seq_len, hidden_size)
    print(enc_output_i.size())
    dec_hid_i = dec_hid_i.permute(1, 2, 0)
    # enc_hid: (batch, hidden_size, 1)
    print(dec_hid_i.size())

    firstMultiRes = enc_output_i.bmm(dec_hid_i)
    #firstMultiRes (batch, seq_len, 1)
    print(firstMultiRes.size())

    firstMultiRes = firstMultiRes.squeeze(2)
    print(firstMultiRes.size())

    #weight (batch, seq_len)
    weight_i = F.softmax(firstMultiRes, dim=1)
    print(weight_i.size())
    #print(weight_i[0,:])

    enc_output_i = enc_output_i.permute(0,2,1)
    #enc_ouput:  (batch, hidden_size, seq_len)
    print("enc_output size", enc_output_i.size())

    weight_i = weight_i.unsqueeze(2)
    # weight (batch, seq_len, 1)
    print("weight size ", weight_i.size())

    attentionRes = enc_output_i.bmm(weight_i)
    # weight (batch, hidden_size, 1)

    print(attentionRes.size())
    attentionRes = attentionRes.squeeze(2)

    print(attentionRes.size())
    print("attention stopped")
    return attentionRes

context_test = attention(enc_output, enc_hid)

#decoder is always one by one cell
#input: one target vector, not a seq   (batch_size, dec_target_dim)
#enc_hid:  encoder hidden   (1, batch, hidden_size)
#enc_ouput: (seq_len, batch,  hidden_size)
def decoderFunctions(oneInput_i, enc_output_i, dec_hid_i):
    print("decoderFunctions starts")
    context = attention(enc_output_i, dec_hid_i)
    print("context size", context.size())
    print("oneInput size", oneInput_i.size())
    combinedInput = torch.cat((oneInput_i, context), dim=1)
    combinedInput = combinedInput.unsqueeze(1)
    combinedInput = combinedInput.permute(1, 0, 2)

    print("combinedInput size ", combinedInput.size())
    print("dec_hid_i size ", dec_hid_i.size())

    rnn = nn.GRU(enc_hidden_dim + dec_target_dim, dec_hidden_dim)
    print("GRU input dim ", enc_hidden_dim + dec_target_dim)
    print("GRU hidden dim ", dec_hidden_dim)
    dec_output_i, dec_hid_i = rnn(combinedInput, dec_hid_i)

    print("dec_output_i size ", dec_output_i.size())
    print("dec_hid_i size ", dec_hid_i.size())

    linear_out1 = nn.Linear(dec_hidden_dim, dec_target_temp_dim)
    linear_out2 = nn.Linear(dec_target_temp_dim, dec_target_dim)

    target_output_temp = F.relu(linear_out1(dec_output_i))
    print("target_output_temp size ", target_output_temp.size())

    target_output = F.relu(linear_out2(target_output_temp))
    print("target_output size ", target_output.size())

    print("decoderFunctions ends")
    return target_output, dec_hid_i

oneInput = torch.randn(batch_size, dec_target_dim)

target_output_single, dec_hid_i_single = decoderFunctions(oneInput, enc_output, enc_hid)

def seq2seqFunctions(batch_input_i, batch_output_i):
    print("seq2seqFunctions starts")

    print("batch_input_i size ", batch_input_i.size())
    print("batch_output_i size ", batch_output_i.size())

    outputs_i = torch.zeros(dec_seq_len, batch_size, dec_target_dim)
    print("outputs_i size ", outputs_i.size())

    encoder_outputs_i, temp_hidden_i = encoderFunctions(batch_input_i)
    print("encoder_outputs_i size ", encoder_outputs_i.size())
    print("encoder_hidden size ", temp_hidden_i.size())

    #get first input as target
    oneInput_i = batch_output_i[0,:,:]
    print("oneInput_i size ", oneInput_i.size())

    for t in range(1, dec_seq_len):
        output_single_i, temp_hidden_i  = decoderFunctions(oneInput_i, encoder_outputs_i, temp_hidden_i)
        outputs_i[t] = output_single_i
        oneInput_i = batch_output_i[t]

        print("currently decoder time t ", t)

    print("outputs_i size ", outputs_i.size())
    print("seq2seqFunctions ends")
    return outputs_i

output_predict = seq2seqFunctions(inputBatchData, outputBatchData)







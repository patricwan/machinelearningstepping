from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector
from keras.layers.recurrent import GRU
from keras.models import model_from_json


def build_model_lstm(input_size, max_out_seq_len, hidden_size):
    
    model = Sequential()
    
    # Encoder(The first LSTM)     
    model.add( LSTM(input_dim=input_size, output_dim=hidden_size, return_sequences=False) )
    model.add( Dense(hidden_size, activation="relu") )
    
    # Use "RepeatVector" to make Encoder output (last time step)a N copy as Decoder N time's input
    model.add( RepeatVector(max_out_seq_len) )
    
    # Decoder(The second LSTM) 
    model.add( LSTM(hidden_size, return_sequences=True) )
    
    # TimeDistributed  to make Dense and Decoder consistent
    model.add( TimeDistributed(Dense(output_dim=input_size, activation="linear")) )
  
    model.compile(loss="mse", optimizer='adam')

    return model

def build_model_gru(input_size, seq_len, hidden_size):
    """Build a sequence to sequence model """
    model = Sequential()
    model.add(GRU(input_dim=input_size, output_dim=hidden_size, return_sequences=False))
    model.add(Dense(hidden_size, activation="relu"))
    model.add(RepeatVector(seq_len))
    model.add(GRU(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim=input_size, activation="linear")))
    model.compile(loss="mse", optimizer='adam')

    return model




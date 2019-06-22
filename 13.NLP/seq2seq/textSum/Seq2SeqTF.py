import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

from TextProcessUtil import *
from Seq2SeqParams import *

class Seq2SeqTF(object):
    def __init__(self, params):
        self.params = params
        return None

    def model_inputs(self):
        '''Create palceholders for inputs to the model'''
        
        self.input_data = tf.placeholder(tf.int32, [None, None], name='input')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        self.summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
        self.max_summary_length = tf.reduce_max(self.summary_length, name='max_dec_len')
        self.text_length = tf.placeholder(tf.int32, (None,), name='text_length')

        return self.input_data, self.targets, self.lr, self.keep_prob, self.summary_length, self.max_summary_length, self.text_length

    def process_encoding_input(self, target_data, vocab_to_int, batch_size):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
        
        ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

        return dec_input

    def encoding_layer(self, rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
        '''Create the encoding layer'''
        
        for layer in range(num_layers):
            with tf.variable_scope('encoder_{}'.format(layer)):
                cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                        input_keep_prob = keep_prob)

                cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                        input_keep_prob = keep_prob)

                enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                        cell_bw, 
                                                                        rnn_inputs,
                                                                        sequence_length,
                                                                        dtype=tf.float32)
        # Join outputs since we are using a bidirectional RNN
        enc_output = tf.concat(enc_output,2)
        
        return enc_output, enc_state

    def training_decoding_layer(self, dec_embed_input, summary_length, dec_cell, initial_state, output_layer, 
                            vocab_size, max_summary_length):
        '''Create the training logits'''
        
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=summary_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        training_helper,
                                                        initial_state,
                                                        output_layer) 

        training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=max_summary_length)
        return training_logits

    def inference_decoding_layer(self, embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                                max_summary_length, batch_size):
        '''Create the inference logits'''
        
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
        
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)
                    
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)
                    
        inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_summary_length)
        
        return inference_logits

    def decoding_layer(self, dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, 
                    max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
        '''Create the decoding cell and attention for the training and inference decoding layers'''
        
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                        input_keep_prob = keep_prob)
        
        output_layer = Dense(vocab_size,
                            kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
        
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                    enc_output,
                                                    text_length,
                                                    normalize=False,
                                                    name='BahdanauAttention')

        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                            attn_mech,
                                                            rnn_size)
        
        initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        #initial_state = tf.contrib.seq2seq.AttentionWrapperState(enc_state[0],
        #                                                               _zero_state_tensors(rnn_size, 
        #                                                                                    batch_size, 
        #                                                                                    tf.float32),
        #                                                        time = 0 ,alignments=None , alignment_history=()) 
        with tf.variable_scope("decode"):
            training_logits = self.training_decoding_layer(dec_embed_input, 
                                                    summary_length, 
                                                    dec_cell, 
                                                    initial_state,
                                                    output_layer,
                                                    vocab_size, 
                                                    max_summary_length)
        with tf.variable_scope("decode", reuse=True):
            inference_logits = self.inference_decoding_layer(embeddings,  
                                                        vocab_to_int['<GO>'], 
                                                        vocab_to_int['<EOS>'],
                                                        dec_cell, 
                                                        initial_state, 
                                                        output_layer,
                                                        max_summary_length,
                                                        batch_size)

        return training_logits, inference_logits

    def seq2seq_model(self,input_data, target_data, keep_prob, text_length, summary_length, max_summary_length, 
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, word_embedding_matrix):
        '''Use the previous functions to create the training and inference logits'''
        
        # Use Numberbatch's embeddings and the newly created ones as our embeddings
        embeddings = word_embedding_matrix
        
        enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
        enc_output, enc_state = self.encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
        
        dec_input = self.process_encoding_input(target_data, vocab_to_int, batch_size)
        dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
        
        training_logits, inference_logits  = self.decoding_layer(dec_embed_input, 
                                                            embeddings,
                                                            enc_output,
                                                            enc_state, 
                                                            vocab_size, 
                                                            text_length, 
                                                            summary_length, 
                                                            max_summary_length,
                                                            rnn_size, 
                                                            vocab_to_int, 
                                                            keep_prob, 
                                                            batch_size,
                                                            num_layers)
        
        return training_logits, inference_logits

    def buildGraph(self,vocab_to_int, word_embedding_matrix):
        # Build the graph
        self.train_graph = tf.Graph()
        # Set the graph to default to ensure that it is ready for training
        with self.train_graph.as_default():
            
            # Load the model inputs    
            input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = self.model_inputs()

            # Create the training and inference logits
            training_logits, inference_logits = self.seq2seq_model(tf.reverse(input_data, [-1]),
                                                            targets, 
                                                            keep_prob,   
                                                            text_length,
                                                            summary_length,
                                                            max_summary_length,
                                                            len(vocab_to_int)+1,
                                                            self.params.rnn_size, 
                                                            self.params.num_layers, 
                                                            vocab_to_int,
                                                            self.params.batch_size,
                                                            word_embedding_matrix)
            
            # Create tensors for the training logits and inference logits
            training_logits = tf.identity(training_logits.rnn_output, 'logits')
            inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
            
            # Create the weights for sequence_loss
            masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

            with tf.name_scope("optimization"):
                # Loss function
                self.cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    targets,
                    masks)

                # Optimizer
                optimizer = tf.train.AdamOptimizer(self.params.learning_rate)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(self.cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                self.train_op = optimizer.apply_gradients(capped_gradients)
        print("Graph is built.")

    def train(self,sorted_summaries_short, sorted_texts_short, summaries_lengths, texts_lengths,vocab_to_int):
        # Train the Model
        learning_rate_decay = 0.95
        min_learning_rate = 0.0005
        display_step = 20 # Check training loss after every 20 batches
        stop_early = 0 
        stop = 3 # If the update loss does not decrease in 3 consecutive update checks, stop training
        per_epoch = 3 # Make 3 update checks per epoch
        update_check = (len(sorted_texts_short)//self.params.batch_size//per_epoch)-1

        update_loss = 0 
        batch_loss = 0
        summary_update_loss = [] # Record the update losses for saving improvements in the model

        checkpoint = "best_model.ckpt" 
        with tf.Session(graph=self.train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            # If we want to continue training a previous session
            #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
            #loader.restore(sess, checkpoint)
            
            for epoch_i in range(1, self.params.epochs+1):
                update_loss = 0
                batch_loss = 0
                for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                        get_batches(sorted_summaries_short, sorted_texts_short, self.params.batch_size,vocab_to_int)):
                    start_time = time.time()
                    _, loss = sess.run(
                        [self.train_op, self.cost],
                        {self.input_data: texts_batch,
                        self.targets: summaries_batch,
                        self.lr: self.params.learning_rate,
                        self.summary_length: summaries_lengths,
                        self.text_length: texts_lengths,
                        self.keep_prob: self.params.keep_probability})

                    batch_loss += loss
                    update_loss += loss
                    end_time = time.time()
                    batch_time = end_time - start_time

                    if batch_i % display_step == 0 and batch_i > 0:
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                            .format(epoch_i,
                                    self.params.epochs, 
                                    batch_i, 
                                    len(sorted_texts_short) // self.params.batch_size, 
                                    batch_loss / display_step, 
                                    batch_time*display_step))
                        batch_loss = 0

                    if batch_i % update_check == 0 and batch_i > 0:
                        print("Average loss for this update:", round(update_loss/update_check,3))
                        summary_update_loss.append(update_loss)
                        
                        # If the update loss is at a new minimum, save the model
                        if update_loss <= min(summary_update_loss):
                            print('New Record!') 
                            stop_early = 0
                            saver = tf.train.Saver() 
                            saver.save(sess, checkpoint)

                        else:
                            print("No Improvement.")
                            stop_early += 1
                            if stop_early == stop:
                                break
                        update_loss = 0
                    
                            
                # Reduce learning rate, but not below its minimum value
                self.params.learning_rate *= learning_rate_decay
                if self.params.learning_rate < min_learning_rate:
                    self.params.learning_rate = min_learning_rate
                
                if stop_early == stop:
                    print("Stopping Training.")
                    break

    
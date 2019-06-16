import collections
import math
import os
import random
import zipfile
import numpy as np

import tensorflow as tf

#read a text file, convert all words to an array []
def readFileToWords(filename):
    data = []
    fi = open(filename, 'r')
    for line in fi:
        tokens = line.split()
        for token in tokens:
            data.append(token)
    fi.close()
    return data

#read a zip file, convert all words inside zip text files to an array []
def readZipToWords(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

#array to wordCount map
def arrayToCountMap(words):
    countMap = collections.Counter(words)
    
    return countMap

# step 1 remove high frequency words to reduce noises. 
def remove_fre_stop_word(words):
    t = 1e-5  # t value
    threshold = 0.90  #threshold of removal

    # words frequency
    int_word_counts = collections.Counter(words)
    total_count = len(words)
    #calculate word frequency
    word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
    #print("word_freqs " , word_freqs)
    #calculate the frequency to be removed.
    prob_drop = {w: 1 - np.sqrt(t / f) for w, f in word_freqs.items()}
    #print("prob_drop " , prob_drop)
    # sampling for words
    train_words = [w for w in words if prob_drop[w] < threshold]

    return train_words

def build_dataset(words):
    vocabulary_size = len(set(words))
    
    count = [['UNK', -1]]
    
    wordsCount = collections.Counter(words)
    
    count.extend(wordsCount.most_common(vocabulary_size - 1))  
    
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0

    data=[dictionary[word]  if  word in dictionary else 0 for word in words]  
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data_index = 0
def generate_batch_cbow(batch_size, bag_window, data):
    global data_index
    span = 2 * bag_window + 1     # [ bag_window target bag_window ]
    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    #print("batch " , batch)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    #print("labels " , labels)
    #deque : double queue
    buffer = collections.deque(maxlen=span)
    #print("buffer " , buffer)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size):
        # just for testing
        buffer_list = list(buffer)
        labels[i, 0] = buffer_list.pop(bag_window)
        batch[i] = buffer_list
        # iterate to the next buffer
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

def generate_batch_skip(batch_size, num_skips, skip_window, data):
    global data_index        # 使用全局变量，意思是在函数里边也能更改其值
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1                # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)   # double direction queue
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):   #i取值0,1,2
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])                       
        data_index = (data_index + 1) % len(data)
    data_index -= 1                     
    return batch, labels

class Vocabulary:
    #fileName: the file used to build the vocabulary
    def __init__(self, fileName):
        words = readFileToWords(fileName)
        train_words = remove_fre_stop_word(words)
        
        self.data, self.count, self.dictionary, self.reverse_dictionary = build_dataset(train_words)
        return None
    
    def __data__(self):
        return self.data
    
    def __dictionary__(self):
        return self.dictionary

class Word2VecTF:
    def __init__(self, vocabulary, batch_size, embedding_size,bag_window, valid_size,valid_window, num_sampled, num_steps):
        print("Start of initialization of Word2VecTF")
        self.vocabulary = vocabulary
        
        self.batch_size = batch_size
        self.embedding_size = embedding_size  # Dimension of the embedding vector.
        self.bag_window = bag_window  # How many words to consider left and right.
        self.valid_size = valid_size  # Random set of words to evaluate similarity on.
        self.valid_window = valid_window  # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.array(random.sample(range(valid_window), valid_size))
        self.num_sampled = num_sampled  # Number of negative examples to sample.
        self.num_steps=num_steps
        
        print("batch_size", self.batch_size)
        print("embedding_size", self.embedding_size)
        print("bag_window", self.bag_window)
        print("valid_size", self.valid_size)
        print("valid_window", self.valid_window)
        print("valid_examples", self.valid_examples)
        print("num_sampled", self.num_sampled)
        print("num_steps", self.num_steps)
        
        return None
        
        
    def buildGraph(self):
        print("Start of initialization of buildGraph")
        graph = tf.Graph()
        
        vocabulary_size = len(set(self.vocabulary.dictionary.keys()))
        
        with graph.as_default():
            # Input data.
            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.bag_window * 2])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
            
            # Variables.
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, self.embedding_size], -1.0, 1.0))
            softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
            
            
            # Model.
            # Look up embeddings for inputs.
            embeds = tf.nn.embedding_lookup(embeddings, self.train_dataset)
            # Compute the softmax loss, using a sample of the negative labels each time.
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, self.train_labels,
                                  tf.reduce_sum(embeds, 1), self.num_sampled, vocabulary_size))
            
            # Optimizer.
            #self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)
            self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)
            
            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.normalized_embeddings = embeddings / norm
            #valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            #similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
            
        return graph

    
    def train(self, graph):
        print("Start of initialization of train")
        
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            print('Initialized all variables')
            average_loss = 0
            
            for step in range(self.num_steps):
                batch_data, batch_labels = generate_batch_cbow(self.batch_size, self.bag_window, self.vocabulary.data)
                
                feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
                _, lossCal = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += lossCal
        
                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0
            self.final_embeddings = self.normalized_embeddings.eval()
            
        return None
    
    def write_to_file(self,outputFilePath):
        fp = open(outputFilePath, 'w', encoding='utf8')
        for k, v in self.vocabulary.reverse_dictionary.items():
            t = tuple(self.final_embeddings[k])

            s = ''
            for i in t:
                i = str(i)
                s += i + " "

            fp.write(v + " " + s + "\n")

        fp.close()
        
        return None

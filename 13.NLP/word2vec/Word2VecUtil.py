import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np


class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None # Path (list of indices) from the root to the word (leaf)
        self.code = None # Huffman encoding

class Vocab:
    def __init__(self, fi):
        vocab_items = []
        vocab_hash = {}
        word_count = 0

        vocab_items, vocab_hash, word_count = self.readFileToWords(fi)

    def sortByItemKey(self,list):
        list = list.sort(key=lambda eachEle : eachEle.count, reverse=True)
        return list

    def printVocabItemsList(self,list):
        for eachElem in list:
            print("each Voca Item word %s count %d"%(eachElem.word, eachElem.count))

        return None

    def readFileToWords(self,fileName):
        fi = open(fileName, 'r')
        vocab_items = []
        vocab_hash = {}
        word_count = 0

        for line in fi:
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)
                    vocab_items.append(VocabItem(token))
                vocab_items[vocab_hash[token]].count += 1
                word_count += 1
                if word_count % 100 == 0:
                    sys.stdout.write("\rReading word %d" % word_count)
                    sys.stdout.flush()
        
        self.sortByItemKey(vocab_items)

        # Update vocab_hash to the correct order
        vocab_hash = {}
        for i, token in enumerate(vocab_items):
            vocab_hash[token.word] = i

        self.vocab_items = vocab_items
        self.vocab_hash = vocab_hash

        return vocab_items, vocab_hash, word_count
    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash
    
    def indices(self, tokens):
        return [self.vocab_hash[token] for token in tokens]

    def encode_huffman(self):
        vocab_size = len(self)
        count = [t.count for t in self] + [1e15] * (vocab_size - 1)

        parent = [0] * (2 * vocab_size - 2)
        binary = [0] * (2 * vocab_size - 2)        

        pos1 = vocab_size - 1
        pos2 = vocab_size
        for i in range(vocab_size - 1):
            # Find min1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1

            # Find min2
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1

            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = vocab_size + i
            parent[min2] = vocab_size + i
            binary[min2] = 1
        
        # Assign binary code and path pointers to each vocab word
        root_idx = 2 * vocab_size - 2
        for i, token in enumerate(self):
            path = [] # List of indices from the leaf to the root
            code = [] # Binary Huffman encoding from the leaf to the root

            node_idx = i
            while node_idx < root_idx:
                if node_idx >= vocab_size: 
                    path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]

            path.append(root_idx)

            # These are path and code from the root to the leaf
            token.path = [j - vocab_size for j in path[::-1]]
            token.code = code[::-1]
        
        print("count  ", count)
        print("parent  ", parent)
        print("binary  ", binary)



#countByDesc array [99,76,23,34,23,6,5,5,2,1]
def buildHuffmanTree(countByDesc):
    vocab_size = len(countByDesc)

    count = [elem for elem in countByDesc] + [1e9] * (vocab_size - 1)

    parent = [0] * (2 * vocab_size - 2)
    binary = [0] * (2 * vocab_size - 2)    

    print("count initial ", count)
    print(" parent initial ", parent)
    print(" binary initial ", binary)

    pos1 = vocab_size - 1
    pos2 = vocab_size
    for i in range(vocab_size - 1):
        # Find min1
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min1 = pos1
                pos1 -= 1
            else:
                min1 = pos2
                pos2 += 1
        else:
            min1 = pos2
            pos2 += 1

        # Find min2
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min2 = pos1
                pos1 -= 1
            else:
                min2 = pos2
                pos2 += 1
        else:
            min2 = pos2
            pos2 += 1

        count[vocab_size + i] = count[min1] + count[min2]
        parent[min1] = vocab_size + i
        parent[min2] = vocab_size + i
        binary[min2] = 1
    
    root_idx = 2 * vocab_size - 2

    paths = []
    codes = []
    for i in range(0, vocab_size):
        path = [] # List of indices from the leaf to the root
        code = [] # Binary Huffman encoding from the leaf to the root

        node_idx = i
        while node_idx < root_idx:
            if node_idx >= vocab_size: 
                path.append(node_idx)
            code.append(binary[node_idx])
            node_idx = parent[node_idx]

        path.append(root_idx)

        paths.append(path)
        codes.append(code)

    return count, parent, binary, paths, codes



def init_net(dim, vocab_size):
    syn0 = np.random.randn(vocab_size, dim)

    syn1 = np.zeros(syn0.shape)

    return syn0, syn1

class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        
        #got an array from vocab 
        norm = sum([math.pow(eachWord.count, power) for eachWord in vocab]) # Normalizing constant
        print("norm for UnigramTable ", norm)

        table_size = 1e7 # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print('Filling unigram table')
        p = 0 # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

        print("unigram table ", table)
        return None

    def sample(self, count):
        #get random numbers(size) in range (0, high) as array
        #then got corresponding values from table
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]

def train_process(fileName,vocab, syn0, syn1, dim, neg, table):
    fi = open(fileName, 'r')

    #windows size
    win = 5

    word_count = 0
    last_word_count = 0
    
    vocab.encode_huffman()

    print("syn0 skip_gram ", syn0)
    print("syn1 skip_gram ", syn1)
    for line in fi:
        sent = vocab.indices(line.split())
        print("sent(index of the word in vocalb) ", sent)

         #for each line// in sent
        for sent_pos, token in enumerate(sent):

            # Randomize window size, where win is the max window size
            random_win_size = np.random.randint(low=1, high=win+1)

            context_start = max(sent_pos - random_win_size, 0)
            context_end = min(sent_pos + random_win_size + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end] 
            print("context:", context)      

            syn0, syn1 = skip_gram(context, token, syn0, syn1, dim, neg, table, vocab)
            print("syn0  skip_gram ", syn0)
            #print("syn1 cbow ", syn1)
            
            #syn0, syn1 = cbow(context, token, syn0, syn1, dim, neg, table, vocab)
            #print("syn0 cbow ", syn0)
            #print("syn1 cbow ", syn1)

    return None


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

#https://www.cnblogs.com/pinard/p/7249903.html
def skip_gram(context, token, syn0, syn1, dim, neg, table,vocab):
    for context_word in context:
        # Init neu1e with zeros
        neu1e = np.zeros(dim)

        # Compute neu1e and update syn1
        if neg > 0:
            classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
        else:
            #print(" path ", vocab[token].path)
            #print(" code ", vocab[token].code)
            classifiers = zip(vocab[token].path, vocab[token].code)

        for target, label in classifiers:
            #print(" classifiers target %s,%s"%(target,label))
            #print("syn0/1 got one row %s, %s"%(syn0[context_word].shape,syn1[target].shape))
            #got only one row of syn0 and one row of syn1
            sa = syn0[context_word]
            sb = syn1[target]
            #print("sa %s sb %s "%(sa, sb))
            print("mean sa %s, mean sb %s"%(np.mean(sa), np.mean(sb)))

            z = np.dot(sa, sb)
            print("z ",z)
            p = sigmoid(z)
            #print("p sigmoid ",p)

            g = (label - p)
            delta1 = g * syn1[target]
            #print("delta1 ", delta1)

            neu1e += g * syn1[target]              # Error to backpropagate to syn0

            delta2 = g * syn0[context_word]
            #print("delta2 ", delta2)
            # ∂L∂xw0=∑i=0neg(yi−σ(xTw0θwi))θwi
            syn1[target] += g * syn0[context_word] # Update syn1
        # Update syn0
        #∂L∂θwi =(yi−σ(xTw0θwi))xw0
        syn0[context_word] += neu1e
    
    return syn0,syn1


def cbow(context, token, syn0, syn1, dim, neg, table,vocab):
    # Compute neu1
    neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
    
    # Init neu1e with zeros
    neu1e = np.zeros(dim)

    # Compute neu1e and update syn1
    if neg > 0:
        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
    else:
        classifiers = zip(vocab[token].path, vocab[token].code)

    print("cbow get classifiers " , classifiers)
    for target, label in classifiers:
        z = np.dot(neu1, syn1[target])
        p = sigmoid(z)
        g = (label - p)
        neu1e += g * syn1[target] # Error to backpropagate to syn0
        syn1[target] += g * neu1  # Update syn1
    # Update syn0
    for context_word in context:
        syn0[context_word] += neu1e

    return syn0,syn1
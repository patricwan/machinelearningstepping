import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
print('TensorFlow Version: {}'.format(tf.__version__))


def readCSVNeededColumns(fileName,droppedColumns):
    reviews = pd.read_csv(fileName)
    # Remove null values and unneeded features
    reviews = reviews.dropna()
    reviews = reviews.drop(droppedColumns, 1)
    reviews = reviews.reset_index(drop=True)
    return reviews


def clean_text(text, contractions, remove_stopwords = True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    # Convert words to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


def cleanReviews(reviews,contractions):
    clean_summaries = []
    for summary in reviews.Summary:
        clean_summaries.append(clean_text(summary, contractions, remove_stopwords=False))
    print("Summaries are complete.")

    clean_texts = []
    for text in reviews.Text:
        clean_texts.append(clean_text(text, contractions))
    print("Texts are complete.")

    return clean_summaries, clean_texts

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1
    return count_dict

def readExistingWordEmbeddings(emdeddingFile):
    embeddings_index = {}
    with open(emdeddingFile, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    return embeddings_index

def calculateMissingRatio(word_counts, embeddings_index):
    missing_words = 0
    threshold = 20

    for word, count in word_counts.items():
        if count > threshold:
            if word not in embeddings_index:
                missing_words += 1
                
    missing_ratio = round(missing_words/len(word_counts),4)*100
    return missing_ratio

def calculateVocaToInt(word_counts, embeddings_index):
    threshold = 20
    vocab_to_int = {} 

    value = 0
    for word, count in word_counts.items():
        if count >= threshold or word in embeddings_index:
            vocab_to_int[word] = value
            value += 1

    # Special tokens that will be added to our vocab
    codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

    # Add codes to vocab
    for code in codes:
        vocab_to_int[code] = len(vocab_to_int)

    # Dictionary to convert integers to words
    int_to_vocab = {}
    for word, value in vocab_to_int.items():
        int_to_vocab[value] = word

    usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

    return vocab_to_int, int_to_vocab


def calculateWordEmbeddingMatrix(vocab_to_int,embeddings_index):
    # Need to use 300 for embedding dimensions to match CN's vectors.
    embedding_dim = 300
    nb_words = len(vocab_to_int)

    # Create matrix with default values of zero
    word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
    for word, i in vocab_to_int.items():
        if word in embeddings_index:
            word_embedding_matrix[i] = embeddings_index[word]
        else:
            # If word not in CN, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            embeddings_index[word] = new_embedding
            word_embedding_matrix[i] = new_embedding
    return word_embedding_matrix

def convert_to_ints(text, word_count, unk_count, vocab_to_int, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


def sumTextToInt(clean_summaries, clean_texts, vocab_to_int):
    # Apply convert_to_ints to clean_summaries and clean_texts
    word_count = 0
    unk_count = 0

    int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count, vocab_to_int)
    int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, vocab_to_int, eos=True)

    unk_percent = round(unk_count/word_count,4)*100

    return int_summaries, int_texts

def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])

def unk_counter(sentence, vocab_to_int):
    '''Counts the number of time UNK appears in a sentence.'''
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count

def sortSummaryTexts(lengths_texts, int_summaries, int_texts, vocab_to_int):
    sorted_summaries = []
    sorted_texts = []
    max_text_length = 84
    max_summary_length = 13
    min_length = 2
    unk_text_limit = 1
    unk_summary_limit = 0

    for length in range(min(lengths_texts.counts), max_text_length): 
        for count, words in enumerate(int_summaries):
            if (len(int_summaries[count]) >= min_length and
                len(int_summaries[count]) <= max_summary_length and
                len(int_texts[count]) >= min_length and
                unk_counter(int_summaries[count], vocab_to_int) <= unk_summary_limit and
                unk_counter(int_texts[count], vocab_to_int) <= unk_text_limit and
                length == len(int_texts[count])
            ):
                sorted_summaries.append(int_summaries[count])
                sorted_texts.append(int_texts[count])
    return sorted_summaries, sorted_texts

def pad_sentence_batch(sentence_batch,vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(summaries, texts, batch_size,vocab_to_int):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch,vocab_to_int))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch,vocab_to_int))
        
        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))
        
        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))
        
        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths














from TextProcessUtil import *
from Seq2SeqParams import *
from Seq2SeqTF import *


reviews = readCSVNeededColumns("../../../../data/nlp/reviews.csv",['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator',
                        'Score','Time'])
print("head" , reviews.head())

contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
    }

clean_summaries, clean_texts = cleanReviews(reviews, contractions)

word_counts = {}

count_words(word_counts, clean_summaries)
count_words(word_counts, clean_texts)

print("Size of Vocabulary:", len(word_counts))

embeddings_index = readExistingWordEmbeddings('../../../../data/nlp/numberbatch-en-17.04b.txt')

missing_ratio = calculateMissingRatio(word_counts, embeddings_index)

vocab_to_int, int_to_vocab = calculateVocaToInt(word_counts, embeddings_index)

word_embedding_matrix = calculateWordEmbeddingMatrix(vocab_to_int, embeddings_index )

int_summaries, int_texts = sumTextToInt(clean_summaries, clean_texts, vocab_to_int)

lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

sorted_summaries, sorted_texts = sortSummaryTexts(lengths_texts, int_summaries, int_texts, vocab_to_int)
print(len(sorted_summaries), len(sorted_texts))

start = 50000
end = start + 50000
sorted_summaries_short = sorted_summaries[start:end]
sorted_texts_short = sorted_texts[start:end]


seq2SeqParams = Seq2SeqParams()
seq2SeqTF = Seq2SeqTF(seq2SeqParams)
seq2SeqTF.buildGraph(vocab_to_int,word_embedding_matrix)
seq2SeqTF.train(sorted_summaries_short, sorted_texts_short, lengths_summaries, lengths_texts,vocab_to_int)





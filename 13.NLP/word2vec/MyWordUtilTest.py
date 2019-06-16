from WordUtil import *


words = readZipToWords("text8.zip")
#print("words ", words)
words = readFileToWords("engTextSmall.txt")
print("words ", words)

countMap = arrayToCountMap(words)
print("countMap " , countMap)

train_words = remove_fre_stop_word(words)
print("train_words" , train_words)

count = [['UNK', -1]]
wordsCount = collections.Counter(words)
vocabulary_size = len(set(words))
mostCommonWords = wordsCount.most_common(vocabulary_size - 1)
print("mostCommonWords", mostCommonWords)

count.extend(mostCommonWords)  
print("extendedWordsCount", count)

#from words array
data, count, dictionary, reverse_dictionary = build_dataset(words)
print("data top5 ", data[:15])
print("count top5" , count[:15])
print("reverse_dictionary top5" , [reverse_dictionary[i] for i in data[:5]])
print("dictionary top5" , [i for i in count[:5]])

batch, labels = generate_batch_cbow(batch_size=5, bag_window=3, data=data)
print("batch cbow generated ", batch)
print("labels cbow generated " , labels)

#batch, labels = generate_batch_skip(batch_size=8, num_skips=4, skip_window=5,data=data)  
#print("batch skip generated ", batch)
#print("labels skip generated " , labels)

for j in range(10):
    batch, labels = generate_batch_skip(batch_size=8, num_skips=4, skip_window=2, data=data)    
    print("batch skip generated ", batch)
    print("labels skip generated " , labels)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
        
vocabulary = Vocabulary(fileName = "engTextSmall.txt")

word2VecTf = Word2VecTF(vocabulary=vocabulary, batch_size=128, embedding_size=300,bag_window=2, valid_size=16,valid_window=100, num_sampled=64, num_steps=100001)
graph = word2VecTf.buildGraph()
word2VecTf.train(graph)
word2VecTf.write_to_file("testWord2VecCbow.txt")







from Word2VecUtil import *
import numpy as np

vocab = Vocab("engTextSmall.txt")

syn0, syn1 = init_net(300, len(vocab))
print("syn0 ", syn0)
print("syn1 ", syn1)

table = UnigramTable(vocab)
randomSample = table.sample(6)
print("random sample ", randomSample)

train_process("engTextSmall.txt",vocab, syn0, syn1,300, 6, table)

#vocab.encode_huffman()

#countByDesc = [99,76,43,34,23,6,5,5,2,1]
#count, parent, binary, paths, codes = buildHuffmanTree(countByDesc)
#print("count  ", count)
#print("parent ", parent)
#print("binary ", binary)
#print("paths ", paths)
#print("codes ", codes)

testMatrix = np.random.randn(5, 3)
#print("original ", testMatrix)
#print("2 ", testMatrix[2])
#print("3 ", testMatrix[3])
#print("dot result ", np.dot(testMatrix[2], testMatrix[3]))


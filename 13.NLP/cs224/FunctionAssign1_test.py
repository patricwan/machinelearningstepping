import numpy as np
import random

from FunctionAssign1_basic import *

test1 = softmax(np.array([3,4,7]))
print("softmax test1", test1)

test2 = softmax(np.array([[156,102, 677],[3,4,7]]))
print("softmax test2", test2)

xIn = np.array([[1, 2], [3, 23], [4,-9]])
fSig = sigmoid(xIn)
gradOut = sigmoid_grad(fSig)
print("fSig", fSig)
print("gradOut",gradOut)


#lamda test
quad_and_gradMy = lambda x: (np.sum(x ** 2), x * 2)
xValue = np.array(1.2)
fxMy, gradMy = quad_and_gradMy(xValue)
print("xValue", xValue)
print("fxMy", fxMy)
print("gradMy", gradMy)

sigmoid_and_grad = lambda x: (np.sum(sigmoid(x)), sigmoid_grad(sigmoid(x)))
checkResult = gradcheck_naive(sigmoid_and_grad, np.arange(-5.0, 5.0, 0.5))   # range test

    
N = 20
dimensions = [10, 5, 10]
data = np.random.randn(N, dimensions[0])   # each row will be a datum
labels = np.zeros((N, dimensions[2]))
for i in range(N):
    labels[i,random.randint(0,dimensions[2]-1)] = 1
    
params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

quad_grad = lambda x: (np.sum(x ** 3), x ** 2 * 3)

t1 = sgd(quad_grad, 3, 0.01, 1000, PRINT_EVERY=100)
print("sgd test 1 result:", t1)

print("normalizeRows", normalizeRows(np.array([[3.0,4.0, 2],[1, 2, 1]])))

#test word2vec
# Interface to the dataset for negative sampling
dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)

def getRandomContext(C):
    tokens = ["i", "want", "to", "play", "ipad"]
    return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
                                         for i in range(2*C)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext

random.seed(31415)
np.random.seed(9265)
dummy_vectors = normalizeRows(np.random.randn(10,3))
dummy_tokens = dict([("i",0), ("want",1), ("to",2),("play",3),("ipad",4)])

print("skipgram:",skipgram("to", 3, ["i", "want", "to", "play", "want", "to"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
print("skipgram:",skipgram("to", 1, ["i", "want"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
print("cbow:",cbow("i", 2, ["i", "want", "to", "i"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
print("cbow:",cbow("i", 2, ["i", "want", "i", "to"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))



#!/usr/bin/env python

import numpy as np
import random

from FunctionAssign1_basic import softmax
from FunctionAssign1_basic import gradcheck_naive
from FunctionAssign1_basic import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    x_square = x**2
    x_square_sum = np.sqrt(x_square.sum(axis=1)).reshape((-1,1))
    x /= x_square_sum
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print(x)
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("Testing normalizeRows pass")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    # V代表词汇个数，d代表输入向量维数
    V,d = outputVectors.shape

    # predicted 是vc，也就是中心词对应的输入向量
    output = softmax(np.dot(predicted,outputVectors.T))
    cost = (-1) * np.log(output[target])

    # 3(a)公式
    gradPred = np.sum(outputVectors * output.reshape((-1,1)),axis=0) - outputVectors[target]

    # 3(b)公式
    onehot = np.zeros(V)
    onehot[target] = 1
    grad = np.dot((output-onehot).reshape(-1,1),predicted.reshape(1,-1))

    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    vc = predicted
    U = outputVectors
    uo = U[target]

    # 3(c)公式6
    cost = np.log(sigmoid(np.dot(uo,vc)))
    for i in range(1,len(indices)):
        uk = U[indices[i]]
        cost += np.log(sigmoid(-np.dot(uk,vc)))
    cost *= -1

    #
    gradPred = (sigmoid(np.dot(uo,vc)) - 1) * uo
    for i in range(1,len(indices)):
        uk = U[indices[i]]
        gradPred -= (sigmoid(-np.dot(uk,vc))-1)*uk

    #
    grad = np.zeros(U.shape)
    for i in range(1,len(indices)):
        index = indices[i]
        uk = U[index]
        grad[index] += (1-sigmoid(-np.dot(uk,vc)))*vc
    grad[target] = (sigmoid(np.dot(uo,vc))-1)*vc

    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    inputIndex = tokens[currentWord]
    predicted = inputVectors[inputIndex]

    cost = 0
    # 正常情况下len(contextWords)应该要等于2C
    for contextWord in contextWords:
        target = tokens[contextWord]
        costTemp,gradPred,grad= word2vecCostAndGradient(predicted,target,outputVectors,dataset)
        cost += costTemp
        gradOut += grad
        gradIn[inputIndex] += gradPred

    # 求predicted，这个比较容易。也不用先把输入词转为one-hot向量，然后再用矩阵乘法。直接在inputVectors里面查找就行了。
    currentWordIndex = tokens.get(currentWord)
    predicted = inputVectors[currentWordIndex]

    for i in range(2*C):
        if i < len(contextWords):
            contextWord = contextWords[i]
            # token是词表，target是contextWord在词表中对应的位置，target是根据predicted预测出来的。
            target = tokens.get(contextWord)
            costTemp, gradPred, grad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
            cost = cost + costTemp
            gradIn[currentWordIndex] = gradIn[currentWordIndex] + gradPred
            gradOut = gradOut + grad
        else:
            break
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    target = tokens[currentWord]

    contextWordsSum = np.zeros(len(inputVectors[0]))
    for contextWord in contextWords:
        index = tokens[contextWord]
        contextWordsSum += inputVectors[index]

    cost, grad, gradOut = word2vecCostAndGradient(contextWordsSum,target,outputVectors,dataset)

    for contextWord in contextWords:
        index = tokens[contextWord]
        gradIn[index] += grad
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    N_half = int(N/2)
    inputVectors = wordVectors[:N_half,:]
    outputVectors = wordVectors[N_half:,:]
    for i in range(batchsize):
        # 邻居长度C1
        C1 = random.randint(1,C)
        # 随机得到一个centerword，和包含2*C1的context
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N_half, :] += gin / batchsize / denom
        grad[N_half:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    # 10个向量会分成5个input向量和5个output向量
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    word2vec_sgd_wrapper(skipgram, dummy_tokens, dummy_vectors, dataset, 5, softmaxCostAndGradient);
    
    
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
     #   skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),   dummy_vectors)
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
     #   skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),  dummy_vectors)
    
    #print("\n==== Gradient check for CBOW      ====")
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
     #   cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient), dummy_vectors)
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #    cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),   dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],  dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    
    #print(skipgram("c", 1, ["a", "b"],
    #    dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"],   dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    #print(cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))


if __name__ == "__main__":
    # test_normalize_rows()
    test_word2vec()

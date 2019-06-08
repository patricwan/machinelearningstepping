import glob
import random
import numpy as np
import os.path as op
from scipy.special import expit

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    The expit function, also known as the logistic function, 
    is defined as expit(x) = 1/(1+exp(-x)). 
    """
    return expit(x)

def sigmoid_grad(f):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input f should be the sigmoid
    function value of your original input x. 
    """
    return f - f * f

def softmax(x):
    """
    Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    """
    assert len(x.shape) <= 2
    y = np.exp(x - np.max(x, axis=len(x.shape) - 1, keepdims=True))
    normalization = np.sum(y, axis=len(x.shape) - 1, keepdims=True)
    
    return np.divide(y, normalization)

def gradcheck_naive(f_and_grad, x):
    """ 
    Gradient check for a function f
    - f_and_grad should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 
    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f_and_grad(x) # Evaluate function value at original point
    
    y = np.copy(x)
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        print("loop x Index and value", ix, x[ix])
        
        reldiff = 1.0
        for negative_log_h in range(2, 10):
            h = 0.5 ** negative_log_h
            y[ix] = x[ix] + h
            random.setstate(rndstate)
            
            fy, _ = f_and_grad(y)
            y[ix] = x[ix]
            
            numgrad = (fy - fx) / h
            if fx != fy:
                reldiff = min(reldiff, abs(numgrad - grad[ix]) / max((1.0, abs(numgrad), abs(grad[ix]))))
        print('reldiff', reldiff)
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            return False
        
        it.iternext()        # Step to next dimension
    
    print("Gradient Check passed")
    return True

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    
    #extract input data
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    
     ### Forward propagation: W1*X + b1 =>sigmoid(h1) => W2*sigmoid(h1) + b2 => h2 => softmax(h2)=> Y
    h_per_item = sigmoid(np.dot(data, W1) + b1)
    yhat_per_item = softmax(np.dot(h_per_item, W2) + b2)
    cost = -np.sum(labels * np.log(yhat_per_item))
    
    #backward propagation
    #difference between real labels and calculated output
    #(Y-labels)
    grad_softmax_per_item = yhat_per_item - labels
    
    #Then calcuate grad of each one
    grad_b2 = np.sum(grad_softmax_per_item, axis=0, keepdims=True)
    #h1.T  . (Y-labels)
    grad_W2 = np.dot(h_per_item.T, grad_softmax_per_item)
    
    #sigmoid(h1) *(1-sigmoid(h1))
    grad_sigmoid_per_item = sigmoid_grad(h_per_item)
    # ((Y-labels) . W2.T) * (sigmoid(h1) *(1-sigmoid(h1)))
    grad_b1_per_item = np.dot(grad_softmax_per_item, W2.T) * grad_sigmoid_per_item
    grad_b1 = np.sum(grad_b1_per_item, axis=0, keepdims=True)
    
    #X.T . grad_b1_per_item
    grad_W1 = np.dot(data.T, grad_b1_per_item)
    
    ### Stack gradients
    grad = np.concatenate((grad_W1.flatten(), grad_b1.flatten(), grad_W2.flatten(), grad_b2.flatten()))
    
    print("from forward_backward_prop: cost ", cost);
    print("from forward_backward_prop:grad", grad);
    return cost, grad

def sgd(f_and_grad, x0, step, iterations, postprocess = None, useSaved = False, PRINT_EVERY=10):
    """ Stochastic Gradient Descent """
    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000
    
    start_iter = 0
    x = x0

    if not postprocess:
        postprocess = lambda x: x
    
    print("sgd input ", x, postprocess)
        
    expcost = None
    
    for iter in range(start_iter + 1, iterations + 1):
        cost, grad = f_and_grad(x)
        
        x = postprocess(x - step * grad)
        
        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print("iter %d: %f" % (iter, expcost))
        
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x

def normalizeRows(x):
    """ Row normalization function """
    return x / np.sqrt(np.sum(x * x, axis=1, keepdims=True))

# - predicted: numpy ndarray, predicted word vector (\hat{v} in 
#   the written component or \hat{r} in an earlier version)
# - target: integer, the index of the target word               
# - outputVectors: "output" vectors (as rows) for all tokens     
# - dataset: needed for negative sampling, unused here.    
def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    #assuming the softmax prediction function and cross entropy loss.     
    scalar_products = np.sum(outputVectors * predicted, axis=1)
    yhat = softmax(scalar_products)
    #cross entropy loss
    cost = -np.log(yhat[target])
    
    #about newaxis https://www.jb51.net/article/144967.htm 
    gradPred = np.sum(outputVectors * yhat[:, np.newaxis], axis=0) - outputVectors[target]
    grad = yhat[:, np.newaxis] * predicted[np.newaxis, :]
    grad[target] = grad[target] - predicted
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient about the predicted word vector                                                
    # - grad: the gradient about all the other word vectors   
    return cost, gradPred, grad
#Input/Output Specifications: same as softmaxCostAndGradient  
def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models """
    
    negative_samples = [dataset.sampleTokenIdx() for k in range(K)]
    cost = -np.log(sigmoid(np.dot(outputVectors[target], predicted))) - \
                        sum(np.log(sigmoid(-np.dot(outputVectors[k], predicted))) \
                         for k in negative_samples)
    gradPred = (sigmoid(np.dot(outputVectors[target], predicted)) - 1.0) * outputVectors[target] + \
                sum((1.0 - sigmoid(-np.dot(outputVectors[k], predicted))) * outputVectors[k]
                    for k in negative_samples)
    grad = np.zeros_like(outputVectors)
    grad[target] += (sigmoid(np.dot(outputVectors[target], predicted)) - 1.0) * predicted
    for k in negative_samples:
      grad[k] += (1.0 - sigmoid(-np.dot(outputVectors[k], predicted))) * predicted
    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    cost = 0.0
    gradIn = np.zeros_like(inputVectors)
    gradOut = np.zeros_like(outputVectors)

    center = tokens[currentWord]
    for context_word in contextWords:
      target = tokens[context_word]
      cur_cost, cur_grad_predicted, cur_grad_out = \
          word2vecCostAndGradient(inputVectors[center], target, outputVectors, dataset)
      cost += cur_cost
      gradIn[center] += cur_grad_predicted
      gradOut += cur_grad_out

    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """
         
    # Input/Output specifications: same as the skip-gram model        
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    center = tokens[currentWord]
    vhat = sum(inputVectors[tokens[j]] for j in contextWords)
    cost, grad_predicted, gradOut = word2vecCostAndGradient(vhat, center, outputVectors, dataset)
    for j in contextWords:
      gradIn[tokens[j]] += grad_predicted

    return cost, gradIn, gradOut

import os.path as op
import pickle as pickle


def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter
    if st > 0:
        print("Loading saved params %d" % st)
        with open("saved_params_%d.npy" % st, "rb") as f:
            params = pickle.load(f,encoding='iso-8859-1')
            state = pickle.load(f,encoding='iso-8859-1')
        return st, params, state
    else:
        return st, None, None


    
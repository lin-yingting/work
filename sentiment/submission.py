#!/usr/bin/python

import random
import copy
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    m = x.split(" ")
    dic = {}
    for i in m:
        if i in dic.keys():
            dic[i] = dic[i]+1
        else:
            dic[i] = 1
    return dic
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    # feature = [featureExtractor(i[0]) for i in trainExamples]
    # print([trainExamples[i][0] for i in range(len(trainExamples))])
    feature = [featureExtractor(trainExamples[i][0]) for i in range(len(trainExamples))]

    for k1 in range(len(trainExamples)):
        for k2 in feature[k1].keys():
            if k2 in weights.keys():
                continue
            else:
                weights[k2] = 0
    for i in range(numEpochs):
        for j in range(len(trainExamples)):
            y = (1 if dotProduct(featureExtractor(trainExamples[j][0]), weights)>=0 else -1)
            if y == trainExamples[j][1]:
                continue
            # sdF = -1*feature[j]*trainExamples[j][1]
            # return sum(d1.get(f, 0) * v for f, v in list(d2.items()))
            for k in feature[j].keys():
                weights[k] = weights[k] - eta * (-feature[j][k]*trainExamples[j][1])
        trainError = evaluatePredictor(
            trainExamples, lambda x:
            (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        validationError = evaluatePredictor(
            validationExamples, lambda x:
            (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
        print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))
    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = {}
        for i in weights.keys():
            if random.random()>0.5:
                continue
            else:
                phi[i] = random.randint(0,1000)
        y = (1 if dotProduct(phi, weights) >= 0 else -1)
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x = x.replace(" ", "")
        m = [x[i:i+n] for i in range(len(x)-n+1)]
        dic = {}
        for i in m:
            if i in dic.keys():
                dic[i] = dic[i] + 1
            else:
                dic[i] = 1
        return dic
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 5: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)
    centers = [{}]*K
    counts = [0]*len(examples)
    for i in examples:
        for j in i.keys():
            if j in centers[0].keys():
                continue
            else:
                for k in range(K):
                    centers[k][j] = random.random()*5

    assignments = [-1]*len(examples)
    beforeAssignment = []
    err = 0.0
    for i in range(maxEpochs):
        for j in range(len(examples)):
            for k in range(K):
                if k == 0:
                    small = sum((centers[k].get(f, 0)-v)**2 for f, v in list(examples[j].items()))
                m = sum((centers[k].get(f, 0) - v) ** 2 for f, v in list(examples[j].items()))
                if small >= m:
                    small = m
                    assignments[j] = k
        for k in range(K):
            centers[k] = {k:0 for k in centers[k].keys()}
        err = 0.0
        counts = [0] * len(examples)
        for j in range(len(assignments)):
            for k in examples[j].keys():
                centers[assignments[j]][k] = centers[assignments[j]][k]+examples[j][k]
            counts[assignments[j]] += 1
        for k in range(K):
            if counts[k] == 0:
                continue
            centers[k] = {f: v/counts[k] for f,v in centers[k].items()}
        for j in range(len(assignments)):
            err = err+sum((centers[assignments[j]].get(f, 0) - v) ** 2 for f, v in list(examples[j].items()))
        print(err)

        if beforeAssignment == assignments:
            break
        beforeAssignment = copy.deepcopy(assignments)
    t = (assignments, centers, err)
    return t

    # END_YOUR_CODE

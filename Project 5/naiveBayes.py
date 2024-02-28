# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        "*** YOUR CODE HERE ***"
        best = -1
        probs = util.Counter()
        conds = util.Counter()
        counts = util.Counter()
        for i in range(len(trainingData)):
            probs[trainingLabels[i]] += 1
            for x,v in trainingData[i].items():
                counts[(x,trainingLabels[i])] += 1
                if v > 0:
                    conds[(x, trainingLabels[i])] += 1
        for k in kgrid:
            tempprobs = util.Counter()
            tempconds = util.Counter()
            tempcounts = util.Counter()
            for x, v in probs.items():
                tempprobs[x] += v
            for x, v in counts.items():
                tempcounts[x] += v
            for x, v in conds.items():
                tempconds[x] += v
            for l in self.legalLabels:
                for x in self.features:
                    tempconds[(x,l)] +=  k
                    tempcounts[(x,l)] +=  2*k
            tempprobs.normalize()
            for x, c in tempconds.items():
                tempconds[x] = c * 1 / tempcounts[x]
            self.probs = tempprobs
            self.conds = tempconds
            check = self.classify(validationData)
            valid = 0
            for i in range(len(validationLabels)):
                if check[i] == validationLabels[i]:valid+=1
            if valid > best:
                bestParams = (tempprobs, tempconds, k)
                best = valid
        self.probs, self.conds, self.k = bestParams

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"
        for y in self.legalLabels:
            logJoint[y] = math.log(self.probs[y])
            for x, value in datum.items():
                if value > 0:
                    logJoint[y] += math.log(self.conds[x,y])
                else:
                    logJoint[y] += math.log(1-self.conds[x,y])

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        for x in self.features:
            prob = self.conds[x,label1]/self.conds[x,label2]
            featuresOdds.append(prob)
        featuresOdds.sort(reverse=True)
        while len(featuresOdds) > 100:
            featuresOdds.pop()
        return featuresOdds

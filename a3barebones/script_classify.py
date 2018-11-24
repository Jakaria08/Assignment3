from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import classalgorithms as algs


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))

## k-fold cross-validation
# K - number of folds
# X - data to partition
# Y - targets to partition
# classalgs - a dictionary mapping algorithm names to algorithm instances
#
# example:
classalgs_fold = {
  #'nn_0.01': algs.NeuralNet({ 'regwgt': 0.01, 'nh': 8, 'stepsize': 0.001, 'epochs': 100 }),
  'NeuralNet': algs.NeuralNet({ 'regwgt': 0.05, 'nh': 16, 'stepsize': 0.01, 'epochs': 1000 }),
  #'nn_0.1':  algs.NeuralNet({ 'regwgt': 0.1, 'nh': 32 , 'stepsize': 0.05, 'epochs': 100 }),
  'Logistic Regression1': algs.LogitReg({'regularizer': 'l2', 'regwgt': 0.05}),
  #'Logistic Regression2': algs.LogitReg({'regularizer': 'l1', 'regwgt': 0.01}),
  #'Logistic Regression3': algs.LogitReg({'regularizer': 'l2', 'regwgt': 0.05}),
   }

parametersc = (
        #{'regwgt': 0.0, 'nh': 4},
        { 'regwgt': 0.05, 'nh': 16, 'stepsize': 0.01, 'epochs': 1000 },
        { 'regwgt': 0.01, 'nh': 8, 'stepsize': 0.01, 'epochs': 1000 },
        #{'regwgt': 0.1, 'nh': 32},
                      )
numparamsc = len(parametersc)
errorsc = {}
errorscT = {}

def cross_validate(K, X, Y, classalgs_fold):
    data_range = (int)(X.shape[0]/K)
    fold = list()
    yfold = list()
    trainset = []
    validationset = []
    ytrain = []
    yvalid = []
    check = None
    for learnername in classalgs_fold:
        errorsc[learnername] = np.zeros((numparamsc,K))

    for i in range(K):
        fold.append(X[i*data_range:data_range*(i+1),:])
        yfold.append(Y[i*data_range:data_range*(i+1)])
        
    for k in range(K):
        check = 1
        for j in range(K):
            if j!= k:
                if check:
                    trainset = fold[k]
                    ytrain = yfold[k]
                    check = False
                else:
                    trainset = np.concatenate((trainset,fold[k]), axis = 0) 
                    ytrain = np.concatenate((ytrain,yfold[k]), axis = 0)
            else:
                validationset = fold[k]
                yvalid = yfold[k]
                
        for p in range(numparamsc):
            params = parametersc[p]
                
            for learnername, learner in classalgs_fold.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset, ytrain)
                # Validate model
                predictions = learner.predict(validationset)
                error = geterror(yvalid, predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errorsc[learnername][p,k] = error
            
    for learnername, learner in classalgs_fold.items():
        besterror = np.mean(errorsc[learnername][0,:])
        bestparams = 0
        for p in range(numparamsc):
            aveerror = np.mean(errorsc[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parametersc[bestparams])
        print ('(Cross Validation) Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('(Cross Validation) Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errorsc[learnername][bestparams,:])/math.sqrt(K)))
                    
        #best_algorithm = classalgs_fold[learnername]
        #return best_algorithm
        
def cross_validate_stratified(K, X, Y, classalgs_fold): #### For 5 fold 
    data_range = ((int)(X.shape[0]/K))-2
    fold = list()
    yfold = list()
    XT = list()
    trainset = []
    validationset = []
    ytrain = []
    yvalid = []
    check = None
    for learnername in classalgs_fold:
        errorscT[learnername] = np.zeros((numparamsc,K))

        Z = np.c_[ X, Y ]  
        idx = Z[:, 9] == 0
        X0 = Z[idx]
        idx = Z[:, 9] == 1
        X1 = Z[idx]

        percentage0 = int(np.rint((X0.shape[0]/X.shape[0])*100))
        percentage1 = int(np.rint((X1.shape[0]/X.shape[0])*100))
        
        for i in range(K):
            XT.append(np.concatenate((X0[i*percentage0:percentage0*(i+1)],X1[i*percentage1:percentage1*(i+1)]),axis = 0))
        
    for k in range(K):
        check = 1
        for j in range(K):
            if j!= k:
                if check:
                    trainset = XT[k][:,0:9]
                    ytrain = XT[k][:,9:10]
                    check = False
                else:
                    trainset = XT[k][:,0:9] 
                    ytrain = XT[k][:,9:10]
            else:
                validationset = XT[k][:,0:9] 
                yvalid =  XT[k][:,9:10]
                
        for p in range(numparamsc):
            params = parametersc[p]
                
            for learnername, learner in classalgs_fold.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset, ytrain)
                # Validate model
                predictions = learner.predict(validationset)
                error = geterror(yvalid, predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errorscT[learnername][p,k] = error
            
    for learnername, learner in classalgs_fold.items():
        besterror = np.mean(errorscT[learnername][0,:])
        bestparams = 0
        for p in range(numparamsc):
            aveerror = np.mean(errorscT[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parametersc[bestparams])
        print ('(Cross Validation Stratified) Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('(Cross Validation Stratified) Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errorscT[learnername][bestparams,:])/math.sqrt(K)))
                    
        #best_algorithm = classalgs_fold[learnername]
        #return best_algorithm


if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 5

    classalgs = {'Random': algs.Classifier(),
                 'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 'Naive Bayes Ones': algs.NaiveBayes({'usecolumnones': True}),
                 'Linear Regression': algs.LinearRegressionClass(),
                 'Logistic Regression': algs.LogitReg({'regularizer': 'l2', 'regwgt': 0.00}),
                 'Neural Network': algs.NeuralNet({'epochs': 1000}),
                 'KernelLogitReg': algs.KernelLogitReg(),
                 'KernelLogitReg1': algs.KernelLogitReg({'kernel': 'hamming'}),
                 'Neural Network Two Hidden Layer': algs.NeuralNetTwoHidden({'epochs': 1000})
                }
    numalgs = len(classalgs)

    parameters = (
        #{'regwgt': 0.0, 'nh': 4},
        #{'regwgt': 0.01, 'nh': 8},
        {'regwgt': 0.05, 'nh': 16},
        #{'regwgt': 0.1, 'nh': 32},
                      )
    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_susy(trainsize,testsize)
        #trainset, testset = dtl.load_susy_complete(trainsize,testsize)
        #trainset, testset = dtl.load_census(trainsize,testsize)

        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.items():
                # Reset learner for new parameters
                if learnername == 'KernelLogitReg1':
                    trainset1, testset1 = dtl.load_census(trainsize,testsize)
                    learner.reset(params)
                    print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                    # Train model
                    learner.learn(trainset1[0], trainset1[1])
                    # Test model
                    predictions = learner.predict(testset1[0])
                    error = geterror(testset1[1], predictions)
                    print ('Error for ' + learnername + ': ' + str(error))
                    errors[learnername][p,r] = error
                else:
                    learner.reset(params)
                    print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                    # Train model
                    learner.learn(trainset[0], trainset[1])
                    # Test model
                    predictions = learner.predict(testset[0])
                    error = geterror(testset[1], predictions)
                    print ('Error for ' + learnername + ': ' + str(error))
                    errors[learnername][p,r] = error
    

    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))
        
################# Cross validation k fold
        
    cross_validate(5, trainset[0], trainset[1], classalgs_fold)
    cross_validate_stratified(5, trainset[0], trainset[1], classalgs_fold)
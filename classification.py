'''
Created on Aug 3, 2016

@author: dicle
'''
import os
import sys
import math
import numpy as np
import random
import datetime

from sklearn import naive_bayes
from sklearn import svm
from sklearn import dummy
from sklearn import cross_validation
from sklearn import preprocessing, metrics

import IOtools, metaexperimentation

''' 
runs the learning algorithms .
records the learning results.
'''

    
'''
-- df is the data matrix with instance and feature labels as well as original class labels and is stored at dfpath. 
   the matrix is read from dfpath, train/test split performed with n-fold cross validation and the resultant vectors are input to the learning algorithms to get the results.
'''
def get_data(dfpath):
    df = IOtools.readcsv(dfpath, keepindex=True)
    
    X = df.iloc[:, :-1]  # till the last col, the matrix is instancesXfeaturevalues
    y = df.iloc[:, -1]   # the last column is for the original labels of the instances.   
    
    X[np.isnan(X)] = 0   # if any feature value is infinite or NaN, then it is assigned 0.
    X[np.isinf(X)] = 0
        
    return X, y




def apply_cross_validated_learning(datasetname, X, y, resultsfolder, nfolds=5):

    dataspacename = datasetname + "_nfolds-" + str(nfolds)
    experimentrootpath = IOtools.ensure_dir(os.path.join(resultsfolder, dataspacename))
    scorefilepath = os.path.join(experimentrootpath, metaexperimentation.scorefilename+".csv")
    metaexperimentation.initialize_score_file(scorefilepath)
    
    # SVM
    kernels = ["linear", "rbf", "sigmoid", "poly"]
    Cs = [1, 10, 100, 1000]
    
    for kernel in kernels:
        for c in Cs:
            
            alg = "SVM"
            modelname = "_m-" + alg + "_k-" + kernel + "_C-" + str(c)
            experimentname = "nfolds-" + str(nfolds) + modelname
            
            clf = svm.SVC(kernel=kernel, C=c)
            ypredicted = cross_validation.cross_val_predict(clf, X, y, cv=nfolds)
            #print metrics.accuracy_score(y, ypredicted)
            reportresults(y, ypredicted, experimentname, experimentrootpath, scorefilepath)
    
    
    # Naive Bayes
    NBmodels = [naive_bayes.MultinomialNB(), naive_bayes.GaussianNB()]
    for nbmodel in NBmodels:
        alg = "NB"
        modelname = "_m-" + nbmodel.__class__.__name__
        experimentname = "nfolds-" + str(nfolds) + modelname
        
        ypredicted = cross_validation.cross_val_predict(nbmodel, X, y, cv=nfolds)
        reportresults(y, ypredicted, experimentname, experimentrootpath, scorefilepath)


def apply_once_validated_learning(datasetname, X, y, resultsfolder):
    
    nfolds = 0
    dataspacename = datasetname + "_nfolds-" + str(nfolds)
    experimentrootpath = IOtools.ensure_dir(os.path.join(resultsfolder, dataspacename))
    scorefilepath = os.path.join(experimentrootpath, metaexperimentation.scorefilename+".csv")
    metaexperimentation.initialize_score_file(scorefilepath)
    
    # SVM
    kernels = ["linear", "rbf", "sigmoid", "poly"]
    Cs = [1, 10, 100, 1000]
    
    Xtrain, Xtest, ytrain, ytest = cross_validation.train_test_split(X, y, test_size=0.2, random_state=42)
    
    for kernel in kernels:
        for c in Cs:
            
            alg = "SVM"
            modelname = "_m-" + alg + "_k-" + kernel + "_C-" + str(c)
            experimentname = "nfolds-" + str(nfolds) + modelname
            
            clf = svm.SVC(kernel=kernel, C=c)
            clf.fit(Xtrain, ytrain)
            ypredicted = clf.predict(Xtest)
            #print metrics.accuracy_score(y, ypredicted)
            reportresults(ytest, ypredicted, experimentname, experimentrootpath, scorefilepath)
            print "finished running the model ", modelname, " ", str(datetime.datetime.now())
    
    
    # Naive Bayes
    NBmodels = [naive_bayes.MultinomialNB(), naive_bayes.GaussianNB()]
    for nbmodel in NBmodels:
        alg = "NB"
        modelname = "_m-" + nbmodel.__class__.__name__
        experimentname = "nfolds-" + str(nfolds) + modelname
        
        nbmodel.fit(Xtrain, ytrain)
        ypredicted = nbmodel.predict(Xtest)
        reportresults(y, ypredicted, experimentname, experimentrootpath, scorefilepath)
        print "finished running the model ", modelname, " ", str(datetime.datetime.now())
        
        
def apply_random_baseline(y, experimentrootpath, scorefilepath):    
    

    experimentname = "baseline_random"
    distinctlabels = list(set(y))
    ypred = [random.choice(distinctlabels) for _ in range(len(y))]
    reportresults(y, ypred, experimentname, experimentrootpath, scorefilepath)
    
    
    experimentname = "baseline_majority"
    labelcount = [y.tolist().count(label) for label in distinctlabels]
    ind = np.argmax(labelcount)
    maxoccurringlabel = distinctlabels[ind]
    ypred = [maxoccurringlabel] * len(y)
    reportresults(y, ypred, experimentname, experimentrootpath, scorefilepath)
    


'''
def apply_random_baseline(datasetname, X, y, resultsfolder, nfolds=5):    
    
    dataspacename = datasetname + "_nfolds-" + str(nfolds)
    experimentrootpath = IOtools.ensure_dir(os.path.join(resultsfolder, dataspacename))
    scorefilepath = os.path.join(experimentrootpath, metaexperimentation.scorefilename+".csv")
    
    # majority baseline
    clf = dummy.DummyClassifier(strategy='most_frequent',random_state=0)

    experimentname = "baseline_majority"
    reportresults(y, ypredicted, experimentname, experimentrootpath, scorefilepath)
    
    # random baseline
    clf = dummy.DummyClassifier(strategy='stratified',random_state=42)

    experimentname = "baseline_random"
    reportresults(y, ypredicted, experimentname, experimentrootpath, scorefilepath)
'''


def reportresults(ytrue, ypred, experimentname, experimentrootpath, scorefilepath):
            
    precision = metrics.precision_score(ytrue, ypred, pos_label=None, average="macro")
    recall = metrics.recall_score(ytrue, ypred, pos_label=None, average="macro")
    f1score = metrics.f1_score(ytrue, ypred, pos_label=None, average="macro")
    accuracy = metrics.accuracy_score(ytrue, ypred)
        
    scoreline = metaexperimentation.CSVSEP.join(map(lambda x : str(x), [experimentname, precision, recall, f1score, accuracy]))
    IOtools.todisc_txt("\n"+scoreline, scorefilepath, mode="a")
    
    modelscorereportpath = os.path.join(experimentrootpath, experimentname+".txt")   
    try:
        labelnames = list(set(ytrue))
        scorereportstr = metrics.classification_report(ytrue, ypred, target_names=labelnames)
    except:
        scorereportstr = "zero division error\n"
    IOtools.todisc_txt(scorereportstr, modelscorereportpath)
    
    # record instances
    instancesfolder = IOtools.ensure_dir(os.path.join(experimentrootpath, "instances"))
    path = os.path.join(instancesfolder, experimentname+".csv")
    iheader = ["ytrue\t ypred"]
    instances = [str(true)+"\t"+str(pred) for (true, pred) in zip(ytrue, ypred)]
    IOtools.todisc_list(path, iheader+instances)   





if __name__ == '__main__':
    
    '''
    inputrootpath = "/home/dicle/Dicle/Projects/heidelberg_LID/experiment_samples30doc15lang/featurespace"
    learningoutputrootpath = "/home/dicle/Dicle/Projects/heidelberg_LID/experiment_samples30doc15lang/learning_output2/"
    datasets = ["features_2gram"]
    nfolds = 5
    for datasetname in datasets:
        dfpath = os.path.join(inputrootpath, datasetname+metaexperimentation.matrixfileextension)
        X, y = get_data(dfpath)       
        apply_cross_validated_learning(datasetname, X, y, learningoutputrootpath, nfolds)
	'''
    inputrootpath = sys.argv[1]
    learningoutputrootpath = sys.argv[2]
    datasetname = sys.argv[3]
    nfolds = int(sys.argv[4]) 
    
    dfpath = os.path.join(inputrootpath, datasetname+metaexperimentation.matrixfileextension)
    X, y = get_data(dfpath)       
    print "finished grabbing the matrix ", str(datetime.datetime.now())
    apply_cross_validated_learning(datasetname, X, y, learningoutputrootpath, nfolds)
    
    
    
    
    

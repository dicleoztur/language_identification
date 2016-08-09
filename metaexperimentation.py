'''
Created on Aug 3, 2016

@author: dicle
'''


import IOtools

CSVSEP = "\t"
matrixfileextension = ".csv"

scorefilename = "algorithms-scores"
scoresheader = ["algorithm", "precision", "recall", "fscore", "accuracy"]


''''
exprfoldername = "experiments"
scoresfoldername = "scores"
perffoldername = "performance"

#experimentsrootpath = os.path.join(metacorpus.learningrootpath2, exprfoldername)
#expscorepath = os.path.join(experimentsrootpath, scoresfoldername)
#expperfpath = os.path.join(experimentsrootpath, perffoldername) 


#trainpercentage = 70.0
validationpercentage = 15.0
testpercentage = 15.0 #20.0  # percentage of test set


def assign_experimentsrootpath(rootpath):
    return os.path.join(rootpath, exprfoldername)
def assign_expscorepath(rootpath):
    return os.path.join(assign_experimentsrootpath(rootpath), scoresfoldername)
def assign_expperfpath(rootpath):
    return os.path.join(assign_experimentsrootpath(rootpath), perffoldername)
def assign_scorefilepath(scorefolder):
    return os.path.join(scorefolder, scorefilename+".csv")

'''



def initialize_score_file(scorefilepath):
    header = CSVSEP.join(scoresheader)
    IOtools.todisc_txt(header, scorefilepath)  
    
    
    
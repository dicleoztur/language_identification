# language_identification
simple language identification system

 >language_identification
  - feature_extraction.py: given a set of documents, path to an empty folder and an integer n (n of n-gram), builds a matrix (numOfDocs X numOfNGramsInAllDocs) that gives the weighted character (byte tokenised) n-gram statistics of texts and records this matrix on the given empty folder. [on shell it requires #3# arguments: python /?/feature_extraction.py Path_To_Documents Path_To_Empty_Folder N] 
  - classification.py: given a learnable matrix (data points with feature values), path to an empty folder, the name of the dataset (please assign the file name of the learnable matrix) and an integer for the number of folds for cross validation, runs 18 different learning models (SVM with 16 different parameter settings and two naive Bayes models) and reports the results on the empty folder. [on the shell it requires 4 parameters: python classification.py Path_To_Matrix Path_To_Empty_Folder FileName_Of_Matrix N]
  - IOtools.py: contains helper functions for i/o operations.
  - metaexperimentation.py: contains some constants and definitions.

#Three 3rd party libraries, scikit-learn, pandas and nltk, are required.

dicle

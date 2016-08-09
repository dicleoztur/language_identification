# -*- coding: utf-8 -*-
'''
Created on Aug 3, 2016

@author: dicle
'''

import os
import sys
import nltk.util as nltkutils
import nltk
import numpy as np
import pandas as pd
import datetime

import IOtools, metaexperimentation

'''
-- extracts the superword character ngrams in a given corpus. 
-- tokenization is held at the byte-level, not at the encoding-specific CHAR level. we still call the resultant tokens characters because non-latin symbols are characters of the respective language.
-- it is superword because word boundaries (spaces between words), if any, are also considered features  
-- collects all the ngram units in a list along with their counts per file
   
input:
 corpuspath: contains the text files (langID_fileno.txt)
 n: number of characters to measure (n of n-gram)
returns:
 a list that contains the counts of n-grams per file, in a structure called ConditionalFreqDist
'''
def get_superword_char_ngram_counts(corpuspath, n):

    filenames = IOtools.getfilenames_of_dir(corpuspath, removeextension=False)
    
    allngramunits = []
    
    for filename in filenames:
        filepath = os.path.join(corpuspath, filename)
        rawtext = IOtools.readtxtfile2(filepath)
        text = rawtext.decode("utf-8")   # decode the raw text to get the bytes as characters - for tokenization purposes
        

        ngramunits = nltkutils.ngrams(text, n) # ngrams function defined in the nltk library takes a sequence and lists the tuples containing n many succeeding items of that sequence. 
                                                # in our case, the input is a string that has bytes (language-specific characters) as items.                  
        
        for ngramunit in ngramunits:
            allngramunits.append((filename, ngramunit))  # store it as file-unit pairs so that the count of units in each file can easily be calculated
            
    
    cfd = nltk.ConditionalFreqDist(allngramunits)   # ConditionalFreqDist outputs a list that contains the tuple {filename : (ngramunit, count) }, i.e., number of times the text in the file <filename> has the textual unit <ngramunit>.
    
    return cfd




'''
-- converts a given (instance, feature, value) list to a matrix M of size number_of_instances X number_of_features where M[i,j] is the value of feature[j] in instance[i] 
input:
 cfd: list of tuples (instance, feature, value)
 normalize: normalize the resultant matrix
returns:
 matrix whose description given in the previous lines.
'''
def cfd_to_df(cfd, normalize=True):
    rownames = cfd.conditions()  # instance ids
    rownames.sort()
    colnames = [] # feature ids
    for instanceid in rownames:
        colnames.extend(list(cfd[instanceid]))
    #print "b: ", len(colnames)
    colnames = list(set(colnames))
    #print "a: ", len(colnames)
    
    matrix = np.empty([len(rownames), len(colnames)], dtype=object)
    for i,instanceid in enumerate(rownames):
        for j,feature in enumerate(colnames):
            matrix[i][j] = cfd[instanceid][feature]
    if normalize:
        matrix = IOtools.normalize_matrix(matrix)   # normalize to weigh the features throughout the instances (from count to rate)
        
    df = pd.DataFrame(matrix, index=rownames, columns=colnames)  # DataFrame is a structure in pandas library, to store table-like information in the form of matrix with annotatable rows and columns 
    return df


'''
-- assigns the input matrix the class value per instance. the resultant matrix is ready to be input to a learning algorithms
input:
 featuredf: the matrix of instances and features in the DataFrame structure containing instance and feature labels as well
returns:
 df: featuredf + class_column. the class values are original languages the texts are written in and is already included in the filenames which are kept as instance labels.
'''
def assign_original_classes(featuredf):
    class_col = "original_class"
    
    df = featuredf.copy()
    nrows, _ = df.shape
    df[class_col] = ["-"]*nrows   # the original_class attribute is the language of the document
    
    for filename in df.index.values.tolist():
        lang = filename.split("_")[0]  # filename is of the form "LANG_fileno". 
        df.loc[filename, class_col] = lang  
    
    return df

'''
-- runs all the functions to read a set of documents in corpuspath and extract features as character-level n-grams.
   stores the values of the extracted features per document instance in a matrix with metadata (dataframe)
   and records this matrix on the recordpath as tsv file. 
'''
def get_learnable_matrix(corpuspath, n, recordpath):
    cfd = get_superword_char_ngram_counts(corpuspath, n)
    print "finished cfd ", str(datetime.datetime.now())
    featuredf = cfd_to_df(cfd)
    print "finished featuredf ", str(datetime.datetime.now())
    df = assign_original_classes(featuredf)
    print "finished label col ", str(datetime.datetime.now())
    IOtools.tocsv(df, recordpath, keepindex=True)
    #IOtools.todisc_df_byrow(recordpath, df, keepIndex=True)

if __name__ == '__main__':
    
    '''
    corpuspath = "/home/dicle/Dicle/Projects/heidelberg_LID/samples30doc15lang/"
    outpath = "/home/dicle/Dicle/Projects/heidelberg_LID/experiment_samples30doc15lang/featurespace4/"
    n = 5
	'''
    corpuspath = sys.argv[1]
    outpath = sys.argv[2]
    n = int(sys.argv[3])   # caution
  
    csvfilename = "features_" + str(n) + "gram" + metaexperimentation.matrixfileextension
    recordpath = os.path.join(outpath, csvfilename) 
    get_learnable_matrix(corpuspath, n, recordpath)
        
       
        

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 03 12:36:47 2015

@author: Sardhendu_Mishra
"""

from __future__ import division
import sys
import numpy as np
import pandas as pd
from collections import OrderedDict
from time import time
import csv
from scipy.spatial import distance
from scipy import *
from scipy.sparse import * 
from scipy.sparse.linalg import eigsh

np.set_printoptions(precision=4)
import math

import config
conf = config.get_config_settings()


def cal_entropy_of_parent(entropy_column):
    distinct_class_labels=np.unique(entropy_column)
    #print distinct_class
    entropy_parent=0
    for class_name in distinct_class_labels:
        print 'class_name is',class_name
        get_index_of_class=np.where(distinct_class_labels==class_name)[0]   # fetch the index
        #print 'get_index_of_class is', get_index_of_class
        prob_class=len(get_index_of_class)/len(distinct_class_labels)  
        print prob_class
        try:
            ent= -prob_class * math.log(prob_class,2)
            print ent
            entropy_parent=entropy_parent+ent
        except ValueError:
            entropy_parent=entropy_parent + 0 
    return entropy_parent



def cal_information_gain(column_name,feature_columns, label_columns, entropy_of_parent):
    m,n=shape(data_train_data)
    distinct_labels=np.unique(label_columns)
    Information_gain={}
    for columns_no in range(0,n):
        data_column=feature_columns[:,columns_no]   
        distinct_categories=np.unique(data_column)
        weighted_entropy=0
        for categories in distinct_categories:
            print 'category name is', categories
            get_index_of_categories=np.where(data_column==categories)[0]
            #print 'gat_index_of_category is',get_index_of_categories
            entropy_cat=0
            '''Calculating Entropy'''
            for class_name in distinct_labels :
                print 'class_name is',class_name
                get_index_of_class=np.where(label_columns==class_name)[0]   # fetch the index
                #print 'get_index_of_class is', get_index_of_class
                count_cat_n_class=len(np.intersect1d(get_index_of_categories, get_index_of_class))    
                    
                prob_class=count_cat_n_class/len(get_index_of_categories)
                
                if columns_no==0:
                    print 'prob_class is', prob_class
                try:
                    ent= -prob_class *math.log(prob_class,2) 
                    entropy_cat=entropy_cat+ent
                    if columns_no==0 :
                        print 'ent val  is', ent
                except ValueError:
                    entropy_cat=entropy_cat + 0 
                    #if columns_no==0 :
                     #   print 'ent val  is', ent
                    #print 'oops one freaking error', sys.exc_info()[0]
        
            prob_of_cat=len(np.where(data_column==categories)[0])/len(data_column)
            
            if columns_no==0 :
                print 'probability of category %s is' %categories, prob_of_cat
                
            weighted_entropy=weighted_entropy+(prob_of_cat*entropy_cat)    
            print 'weighted_entropy val is', weighted_entropy
                    
        Information_gain[column_name[columns_no]]=entropy_of_parent-weighted_entropy 
        #print  'Information_gain is' , Information_gain          
    
    return    Information_gain,distinct_labels,distinct_categories 
    


#==============================================================================
#  Cal the methods    
#==============================================================================

t1=time() 
data_train=pd.DataFrame()


data_train = pd.read_csv(conf['dataset_all'])

#data_train.columns = [i for i in range(0,data_train.shape[1])]

# 'Test the information gain with different columns'
root_column_name='Price_range'
data_train_data=data_train[data_train.columns[7:15]].copy()  # 7,15 are static for the dataset, however, it may change for other dataset
data_train_class=data_train[root_column_name].copy()

data_train_data = data_train_data.drop(root_column_name, 1)
column_names=data_train_data.columns
  

 
data_train_data=data_train_data.reset_index().values[:,1:]  # 1 is give to exclude the line number
data_train_class=data_train_class.reset_index().values[:,1:]  # 1 is give to exclude the line number

   
#data_train_data=np.array([['steep','bumpy','yes'],['steep','smooth','yes'],['flat','bumpy','no'],['steep','smooth','no']], dtype='object')
#data_train_class=np.array([['slow'],['slow'],['fast'],['fast']], dtype='object')


entropy_of_parent = cal_entropy_of_parent(data_train_class)

Information_gain, distinct_labels, distinct_categories = cal_information_gain(column_names ,data_train_data, data_train_class,entropy_of_parent)

        

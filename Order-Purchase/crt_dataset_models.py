# -*- coding: utf-8 -*-
"""
Created on Thu May 07 18:21:46 2015

@author: Sardhendu_Mishra
"""

from __future__ import division
import itertools
import numpy as np
import csv



import config
conf = config.get_config_settings()

#==============================================================================
# Weighted feature Probability model
#==============================================================================
def build_dataset_jaccard (probability_thresh, data_train, tot_len):
    data_set = [set(row) for row in data_train]
    # now rows are stored as sets; btw list comprehensions are fast.
    flattened_upper_triangle_of_the_matrix = []
    
    for row1, row2 in itertools.combinations(data_set, r=2):
        intersection_len = row1.intersection(row2)
        union_len = len(row1) + len(row2) - len(intersection_len)
        similarity = len(intersection_len) / union_len
        if similarity >= probability_thresh:
            flattened_upper_triangle_of_the_matrix.append(similarity)
        else:
            flattened_upper_triangle_of_the_matrix.append(0)
    return flattened_upper_triangle_of_the_matrix




#==============================================================================
# Get the common ID's count among the canopies and store it in a csv for analysis
#==============================================================================

"""
   Note: This method is not actually used in the algorithm implementation but is created
         and an output file is stored in the disk so that the user can have a good \
         understanding on the intersection and commoness in the dataset.
"""

# create a dict
def count_similarity_among_ids (similarity_dict): 
    cnt_similar_ids_dict={}
    common_cnt=0
    
    for keys, values in similarity_dict.items():
        cnt_similar_ids_dict[keys]={}
        #if keys==1100:
        for ids in range(0,len(values)):
            #print values[ids]
            list_in_orig_id=similarity_dict[keys]
            list_in_cmpr_id= similarity_dict[values[ids]]
            common_cnt=len(set(list_in_orig_id).intersection(list_in_cmpr_id))       
            
            cnt_similar_ids_dict[keys][values[ids]]=common_cnt
                
    # STORE THE SIMILARITY DICTIONARY WITH NUMBER OF IDS SIMILAR IN A CSV FILE 
    with open(conf['similarity_dict'], "wb") as f:  # Just use 'w' mode in 3.x    
        w = csv.writer(f)
        for key, value in cnt_similar_ids_dict.items():
            w.writerow([key, value])
    f.close()           
    return cnt_similar_ids_dict    


     

               
#==============================================================================
#  Now we find the opportune centroids using our difinitive approach
#==============================================================================
def find_opportune_centroids (similarity_dict,matrix_for_cluster):
    centroids_coordinate={}
    excluded_id_lists=[]
    centroid_index=[]
    #index=0
    for keys, values in similarity_dict.items():
        
        if keys in excluded_id_lists:  
            #index=index+1 
            continue
        else:    
            centroid_index.append(keys)
            centroids_coordinate[keys]=matrix_for_cluster[keys]
            # The below code is aimed to extract the last two digits of the ID 
            # For Example for 1101 the below code will give 01            
            excluded_id_lists=excluded_id_lists+values
            #index=index+1 
            
    return  centroid_index  ,centroids_coordinate   
        
        
        

#==============================================================================
#  Getting the best centroids out of the given centroids   
#==============================================================================
   
def find_best_centroids(centroids_ids,centroids_coordinate, cluster_number):
    centroids_coordinate_cmpr=centroids_coordinate.copy()
    count_centroids=len(centroids_coordinate)
    print count_centroids
    centroids_matrix=np.zeros((count_centroids,count_centroids), dtype="float64")
    
    for centroid_index in range(0,count_centroids):
        for centroids_index_cmpr in range(0,count_centroids):
            centroids_matrix[centroid_index][centroids_index_cmpr] = np.linalg.norm(centroids_coordinate[centroid_index]-centroids_coordinate_cmpr[centroids_index_cmpr])
    
    centroids_matrix_sum=np.array([sum(i) for i in zip(*centroids_matrix)], dtype="float64")
    best_centroids_index=centroids_matrix_sum.argsort()[-cluster_number:][::-1]
    
    centroids_best_coordinates=np.array([centroids_coordinate[i] for i in best_centroids_index ], dtype="float64")
    centroids_best_ids=np.array([centroids_ids[i] for i in best_centroids_index ], dtype="int64")
    return   centroids_best_ids,centroids_best_coordinates          




#==============================================================================
# Call the clustering algorithm
#==============================================================================

"""
Notes:
     1> We are levaraging canopy as each training set is going through the centroids chosen
     2> But centroids are chosen wisely, therefore there is no need to run the clustering algorithm 
        for many different random points and hence reduce the complexity.
"""


def cluster_k_means(centroid_index,centroid_coordinates,similarity_dict,matrix_for_cluster):
    flag=1
    matrix_common_centroid_link={}
    while True:
        #print "äääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääää"
        sum_cluster_centroids_dict={}
        len_cluster_centroid_dict={}
        labels={}
        centroid_coordinates_cmpr=centroid_coordinates.copy()
        
        for rows in range(0,len(matrix_for_cluster)):
            current_row=matrix_for_cluster[rows]
            euclidean_dis_array=[]
    
            if flag==1:  
                #print "I am in the if part"
                common_centroids=list(set(similarity_dict[rows]) & set(centroid_index))  # performs intersection
                matrix_common_centroid_link[rows]=common_centroids
                
                for centroids_index in common_centroids:
                    dist = np.linalg.norm(current_row-centroid_coordinates[centroids_index])
                    euclidean_dis_array.append(dist)
                    
                index_min_pos=np.where(euclidean_dis_array==min(euclidean_dis_array))[0][0]
                    
                matrix_array=(current_row).tolist()  #data_train.values[rows]
                nearest_centroid_index=common_centroids[index_min_pos]
                
                if nearest_centroid_index in sum_cluster_centroids_dict: 
                    #print sum_cluster_centroids_dict[nearest_centroid_index],                  
                    sum_cluster_centroids_dict[nearest_centroid_index]=[a+b for a,b in zip(sum_cluster_centroids_dict[nearest_centroid_index],matrix_array)]
                    len_cluster_centroid_dict[nearest_centroid_index] =len_cluster_centroid_dict[nearest_centroid_index] + 1                
                    labels[nearest_centroid_index].append(rows)
                else:
                    sum_cluster_centroids_dict[nearest_centroid_index]=matrix_array
                    len_cluster_centroid_dict[nearest_centroid_index] = 1
                    labels[nearest_centroid_index]=[rows]
                
            else:  
                #print "I am in the else part"
                current_centroids=matrix_common_centroid_link[rows]
                for centroids_index in current_centroids:
                    dist = np.linalg.norm(current_row-centroid_coordinates[centroids_index])  # will calculate the euclidean distance
                    euclidean_dis_array.append(dist)
                    
                index_min_pos=np.where(euclidean_dis_array==min(euclidean_dis_array))[0][0]
     
                matrix_array=(current_row).tolist()  #data_train.values[rows]
                nearest_centroid_index=current_centroids[index_min_pos]
                
                if nearest_centroid_index in sum_cluster_centroids_dict:  
                    #print sum_cluster_centroids_dict[nearest_centroid_index]                 
                    sum_cluster_centroids_dict[nearest_centroid_index]=[a+b for a,b in zip(sum_cluster_centroids_dict[nearest_centroid_index],matrix_array)]
                    len_cluster_centroid_dict[nearest_centroid_index] =len_cluster_centroid_dict[nearest_centroid_index] + 1                                
                    labels[nearest_centroid_index].append(rows)
                else:
                    #print sum_cluster_centroids_dict[nearest_centroid_index] 
                    sum_cluster_centroids_dict[nearest_centroid_index]=matrix_array
                    len_cluster_centroid_dict[nearest_centroid_index] = 1                
                    labels[nearest_centroid_index]=[rows]
                  
                    
        for keys, values in sum_cluster_centroids_dict.items():    
            centroid_coordinates[keys]=np.divide(sum_cluster_centroids_dict[keys],len_cluster_centroid_dict[keys])        
                
        try:
            if np.testing.assert_equal(centroid_coordinates,centroid_coordinates_cmpr)==None:
                break  
        except AssertionError:
            None
            
        flag=flag+1    
    return labels,centroid_coordinates





#==============================================================================
#  Now we have the labels, lets find the clustered rows from data_train
#==============================================================================
def build_the_cluster(labels, data_train) :   
    cluster=[]
    for keys, values in labels.items():
        each_cluster_list=[]
        for indexes in values:
            each_cluster_list.append((data_train.values[indexes]).tolist())
        
        cluster.append(each_cluster_list)
        
    for i in range(0, len(cluster)):    
        cluster[i] = np.vstack([["BU_DESC","CUSTOMER_NAME","Processor","PROD_LN_DESC","LOB_DESC","Price","Is_touch","Flippable_Monitor","disk_memory","Customer_age","Customer_gender","Product_type"],cluster[i]])
    
    return cluster    
   
   


   
   
#==============================================================================
#  Calculate the cost of the chosen centroids   
#==============================================================================
def cal_cost_function_distance (labels,centroid_coordinates,matrix_for_cluster):
    total_distance=0
    for keys, values in labels.items():
        
        K_coordinate=centroid_coordinates[keys]
        for indexes in values:
            C_i=matrix_for_cluster[indexes].tolist()
            
            # Now we find the euclidean distance of all the cluster points from its centroid and add them
            dist = np.linalg.norm(K_coordinate-C_i)
            total_distance=total_distance+dist

    return total_distance
   

   
   
   
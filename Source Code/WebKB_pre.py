# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from sklearn.model_selection import train_test_split
path = './data/WebKB/'
dataset = 'WebKB'
print('Loading {} dataset...'.format(dataset))
def WebKB_preprocessing(path, dataset):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.str)
    dict_ = {}
    

    url_list = list(idx_features_labels[:,0])
    print("number of unique webpages ID: {}".format(len(url_list)))
    for i in range(len(url_list)):
        dict_[url_list[i]] = i
    idx_cp = np.copy(idx_features_labels)
    edges_cp = np.copy(edges_unordered)
    for old, new in dict_.items():
        idx_cp[idx_features_labels == old] = new
        edges_cp[edges_unordered == old] = new
    idx_result = np.asarray(idx_cp[:,:-1],dtype = np.int32)
    # idx_result = np.asarray(idx_result, dtype = np.int32)
    edges_result = np.asarray(edges_cp,dtype = np.int32)
    # edges_result = np.asarray(edges_result, dtype = np.int32)
    return idx_result, edges_result


def data_split(idx, edges, path, dataset):
    # train:0.4 test: 0.5 valid:0.1
    edges_train_set, edges_train_valid_set = train_test_split(edges_result, test_size = 0.6, random_state = 42)

    
    np.savetxt('{}_train_0.4.cites'.format(dataset),edges_train_set, fmt = '%i')
    # np.savetxt('{}_test_0.4.cites'.format(dataset),edges_test_set, fmt = '%i')
    # np.savetxt('{}_valid_0.4.cites'.format(dataset),edges_valid_set, fmt = '%i')
    
    np.savetxt('{}_train_0.4.content'.format(dataset),idx_result, fmt = '%i')
    # np.savetxt('{}_test_0.4.content'.format(dataset),idx_result, fmt = '%i')
    # np.savetxt('{}_valid_0.4.content'.format(dataset),idx_result, fmt = '%i')

idx_result, edges_result = WebKB_preprocessing(path, dataset)

idx_result = np.asarray(idx_result, dtype = np.int32)
edges_result = np.asarray(edges_result, dtype = np.int32)


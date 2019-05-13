
# coding: utf-8

# In[2]:

import time
from lifelines.utils import concordance_index 
import sys
from torch import nn
import numpy as np
import pandas as pd
import network
from torch.utils.data import TensorDataset, Dataset
import torch.utils.data.dataloader as dataloader
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pickle as pkl
from sklearn.preprocessing import StandardScaler


# In[3]:

def printResults(filename,hyperparam_file):
    output=None
    hyperparams=None 
    with open(filename, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        output = u.load()
    with open(hyperparam_file, 'rb') as q:
        u = pkl._Unpickler(q)
        u.encoding = 'latin1'
        hyperparams = u.load()

    
    linear_h = []
    linear_s = []
    nlinear_h = []
    nlinear_s = []
    linear_h_ = []
    linear_s_ = []
    nlinear_h_ = []
    nlinear_s_ = []

    for i in range(len(output)):
        if len(hyperparams[i][2])==1:
            linear_s.append(np.mean(output[i]['c-index-test-soft']))
            linear_h.append(np.mean(output[i]['c-index-test-hard']))
            linear_h_.append(hyperparams[i])
            linear_s_.append(hyperparams[i])
        else:
            nlinear_s.append(np.mean(output[i]['c-index-test-soft']))
            nlinear_h.append(np.mean(output[i]['c-index-test-hard']))
            nlinear_h_.append(hyperparams[i])
            nlinear_s_.append(hyperparams[i])

    print ("Lin-S:", np.max(linear_s), linear_s_[np.argmax(linear_s)])
    print ("Lin-H:", np.max(linear_h), linear_h_[np.argmax(linear_h)])
    print ("NLin-S:", np.max(nlinear_s),nlinear_s_[np.argmax(nlinear_s)])
    print ("NLin-H:", np.max(nlinear_h), nlinear_h_[np.argmax(nlinear_h)])


# In[5]:

printResults('SUPPORT_results.pkl','SUPPORT_hyperparams.pkl')


# In[4]:

printResults('GBSG_results.pkl','GBSG_hyperparams.pkl')


# In[6]:

printResults('METABRIC_results.pkl','METABRIC_hyperparams.pkl')


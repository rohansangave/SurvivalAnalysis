
# coding: utf-8

# In[1]:

import torch
import time
from lifelines.utils import concordance_index 
import sys
from torch import nn
import numpy as np
import pandas as pd
#import network
from torch.utils.data import TensorDataset, Dataset
import torch.utils.data.dataloader as dataloader
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
#get_ipython().magic('matplotlib inline')
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import _pickle as pkl

from torch.multiprocessing import Pool


# In[2]:

print("Dependencies loaded")


# In[3]:

# dataset_dict={1:"WHAS",2:"GBSG",3:"METABRIC",4:"SUPPORT"}
# DATASET_CHOICE = 1 # sys.argv[1]
# dataset_name=dataset_dict[DATASET_CHOICE]

def get_dataset(DATASET_CHOICE):
    if (DATASET_CHOICE == 1):
        # WHAS
        ds = pd.read_csv('./datasets/whas1638.csv',sep=',')
        train = ds[:1310]
        valid = train[-100:]
        train = train[:-100]
        test = ds[1310:]
        x_train = train[['0','1', '2', '3', '4', '5']].as_matrix()
        x_valid = valid[['0','1', '2', '3', '4', '5']].as_matrix()
        x_test = test[['0','1', '2', '3', '4', '5']].as_matrix() 
        name="WHAS"
    elif (DATASET_CHOICE == 2):
        # GBSG
        ds = pd.read_csv('./datasets/gbsg2232.csv',sep=',')
        train = ds[:1546]
        valid = train[-100:]
        train = train[:-100]
        test = ds[1546:]
        x_train = train[['0','1', '2', '3', '4', '5', '6']].as_matrix()
        x_valid = valid[['0','1', '2', '3', '4', '5', '6']].as_matrix()
        x_test = test[['0','1', '2', '3', '4', '5', '6']].as_matrix() 
        name="GBSG"
    elif (DATASET_CHOICE == 3):
        # for METABRIC
        ds = pd.read_csv('./datasets/metabric1904.csv',sep=',')
        train = ds[:1523]
        valid = train[-100:]
        train = train[:-100]
        test = ds[1523:]
        x_train = train[['0','1', '2', '3', '4', '5', '6', '7', '8']].as_matrix()
        x_valid = valid[['0','1', '2', '3', '4', '5', '6', '7', '8']].as_matrix()
        x_test = test[['0','1', '2', '3', '4', '5', '6', '7', '8']].as_matrix() 
        name="METABRIC"
    elif (DATASET_CHOICE == 4):
        # for SUPPORT
        ds = pd.read_csv('./datasets/support8873.csv',sep=',')
        train = ds[:7098]
        valid = train[-100:]
        train = train[:-100]
        test = ds[7098:]
        x_train = train[['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', ]].as_matrix()
        x_valid = valid[['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']].as_matrix()
        x_test = test[['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']].as_matrix()
        name="SUPPORT"
    return x_train,train,x_valid,valid,x_test,test,name


# In[4]:

def scale_data_to_torch(x_train,train,x_valid,valid,x_test,test):
    scl = StandardScaler()
    x_train = scl.fit_transform(x_train)
    e_train = train['fstat']
    t_train = train['lenfol']

    x_valid = scl.fit_transform(x_valid)
    e_valid = valid['fstat']
    t_valid = valid['lenfol']

    x_test = scl.transform(x_test)
    e_test = test['fstat']
    t_test = test['lenfol']

    x_train = torch.from_numpy(x_train).float()
    e_train = torch.from_numpy(e_train.as_matrix()).float()
    t_train = torch.from_numpy(t_train.as_matrix())

    x_valid = torch.from_numpy(x_valid).float()
    e_valid = torch.from_numpy(e_valid.as_matrix()).float()
    t_valid = torch.from_numpy(t_valid.as_matrix())


    x_test = torch.from_numpy(x_test).float()
    e_test = torch.from_numpy(e_test.as_matrix()).float()
    t_test = torch.from_numpy(t_test.as_matrix())
    
    return x_train,e_train,t_train,x_valid,e_valid,t_valid,x_test,e_test,t_test 


# In[5]:

def compute_risk_set(t_train,t_valid,t_test):
    t_ = t_train.cpu().data.numpy()
    risk_set = []
    for i in range(len(t_)):

        risk_set.append([i]+np.where(t_>t_[i])[0].tolist())

    t_ = t_valid.cpu().data.numpy()

    risk_set_valid = []
    for i in range(len(t_)):

        risk_set_valid.append([i]+np.where(t_>t_[i])[0].tolist())


    t_ = t_test.cpu().data.numpy()

    risk_set_test = []
    for i in range(len(t_)):

        risk_set_test.append([i]+np.where(t_>t_[i])[0].tolist())
    return risk_set,risk_set_valid,risk_set_test


# In[6]:

def elbo(risk, gated_output, E, risk_set):
    go_sm = nn.Softmax(dim=1)(gated_output)
    lnumerator = torch.mul(go_sm, risk)
    lnumerator = torch.sum(lnumerator, dim=1)
    expected_risks = torch.exp(risk) * go_sm
    expected_risks = torch.sum(expected_risks, dim=1)
    ldenominator = []
    for i in range(risk.shape[0]):
        ldenominator.append(torch.sum(expected_risks[risk_set[i]], dim=0))
    ldenominator = torch.stack(ldenominator, dim=0)
    ldenominator = torch.log(ldenominator)
    likelihoods = lnumerator - ldenominator
    E =  np.where(E.cpu().data.numpy()==1)[0]
#     neg_likelihood = - torch.sum(likelihoods[E])
    likelihoods = likelihoods[E]
    neg_likelihood = - torch.sum(likelihoods)
    
    return neg_likelihood


# In[7]:

def get_concordance_index(x, gated_x, t, e, bootstrap=False):
    t = t.detach().cpu().numpy()
    e = e.detach().cpu().numpy()
    softmax = torch.nn.Softmax(dim=1)(gated_x)
    r = x.shape[0]
    soft_computed_hazard = torch.exp(x)
    hard_computed_hazard = soft_computed_hazard[range(r),gated_x.argmax(1)[1]]
    soft_computed_hazard = torch.mul(softmax, soft_computed_hazard)
    soft_computed_hazard = torch.sum(soft_computed_hazard, dim = 1)
    soft_computed_hazard = -1*soft_computed_hazard.detach().cpu().numpy()
    hard_computed_hazard = -1*hard_computed_hazard.detach().cpu().numpy()
    if not bootstrap:
        return concordance_index(t,soft_computed_hazard,e),concordance_index(t,hard_computed_hazard,e)
    else:
        soft_concord, hard_concord = [], []
        for i in range(bootstrap):
            soft_dat_, e_, t_ = resample(soft_computed_hazard, e, t,random_state=i )       
            sci = concordance_index(t_,soft_dat_,e_)
            hard_dat_,  e_, t_  = resample(hard_computed_hazard,  e, t ,random_state=i)
            hci = concordance_index(t_,hard_dat_,e_)
            soft_concord.append(sci)
            hard_concord.append(hci)
        return soft_concord, hard_concord


# In[8]:

def train_model(gated_network, betas_network, risk_set, x_train, e_train, t_train, risk_set_valid, x_valid, e_valid, t_valid, 
          optimizer, n_epochs,x_test,e_test,t_test,risk_set_test):
    # Initialize Metrics
    c_index_soft = []
    c_index_hard = []
    train_loss = []
    valid_loss = []
    test_loss = []
    test_c_index_soft = []
    test_c_index_hard = []
    diff = 1e-4
    prev_loss_train = 0
    prev_loss_valid = 0
    bad_cnt = 0
    start = time.time()
    for epoch in range(n_epochs):
        gated_network.train()
        betas_network.train()
        optimizer.zero_grad()
        gated_outputs = gated_network(x_train)
        lsoftmax = nn.LogSoftmax(dim=1)(gated_outputs)
        betas_output = betas_network(x_train)
        ci_train_soft,ci_train_hard = get_concordance_index(betas_output, gated_outputs, t_train, e_train, bootstrap=False)
        c_index_soft.append(ci_train_soft)
        c_index_hard.append(ci_train_hard)
#         loss = negative_log_likelihood(gated_outputs, betas_output, e, risk_set, CUDA)
        loss = elbo(betas_output, gated_outputs, e_train, risk_set) + (betas_network[0].weight**2).sum()
        loss.backward()
        optimizer.step()
        my_loss = loss.cpu().data.numpy()
        train_loss.append(my_loss)
        if abs(my_loss - prev_loss_train) < diff:
            break
        prev_loss_train = my_loss
        torch.cuda.empty_cache()         
        ################################################# Validation #######################################################
        gated_network.eval()
        betas_network.eval()
        gated_outputs_valid = gated_network(x_valid)
        lsoftmax_valid = nn.LogSoftmax(dim=1)(gated_outputs_valid)
        betas_output_valid = betas_network(x_valid)
#         loss = negative_log_likelihood(gated_outputs, betas_output, e, risk_set, CUDA)
        loss_valid = elbo(betas_output_valid, gated_outputs_valid, e_valid, risk_set_valid)
        my_loss_valid = loss_valid.cpu().data.numpy()
        valid_loss.append(my_loss_valid)
        if my_loss_valid - prev_loss_valid > diff:
            bad_cnt+=1
            if bad_cnt>2:
                break
        else:
            bad_cnt=0
        prev_loss_valid = my_loss_valid
        torch.cuda.empty_cache()         

    ################################################# Test #############################################################
    gated_network.eval()
    betas_network.eval()
    gated_outputs_test = gated_network(x_test)
    lsoftmax_test = nn.LogSoftmax(dim=1)(gated_outputs_test)
    betas_output_test = betas_network(x_test)
    ci_test_soft,ci_test_hard = get_concordance_index(betas_output_test, gated_outputs_test, t_test, e_test, bootstrap=250)
    test_c_index_soft.append(ci_test_soft)
    test_c_index_hard.append(ci_test_hard)
#         loss = negative_log_likelihood(gated_outputs, betas_output, e, risk_set, CUDA)
    loss_test = elbo(betas_output_test, gated_outputs_test, e_test, risk_set_test)
    my_loss_test = loss_test.cpu().data.numpy()
    test_loss.append(my_loss_test)
    torch.cuda.empty_cache()         
    
    print('Finished training with %d epochs in %0.2fs' % (epoch + 1, time.time() - start))
    metrics = {}
    metrics['train_loss'] = train_loss
    metrics['valid_loss'] = valid_loss
    metrics['c-index-soft'] = c_index_soft
    metrics['c-index-hard'] = c_index_hard
    metrics['test_loss'] = test_loss
    metrics['c-index-test-soft'] = test_c_index_soft
    metrics['c-index-test-hard'] = test_c_index_hard
    return metrics


# In[9]:

def run_experiment(params):
    linear_model, learning_rate, layers_size, seed,data_dict= params
	print("in")
    layers_size = layers_size[:]
    layers_size += [linear_model]
    torch.manual_seed(seed)
    n_in = data_dict["x_train"].shape[1]
    betas_network = nn.Sequential(nn.Linear(n_in, linear_model, bias=False) )
    layers = []
    for i in range(len(layers_size)-2):
        layers.append(nn.Linear(layers_size[i],layers_size[i+1],bias=False ))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layers_size[-2], layers_size[-1], bias=False))
    gated_network = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(list(gated_network.parameters()) + list(betas_network.parameters()), lr=learning_rate)
#     print(layers)
#     if CUDA:
#         gated_network.cuda()
#         betas_network.cuda()
    n_epochs = 2
    metrics = train_model(gated_network, betas_network, 
                          data_dict["risk_set"], data_dict["x_train"], data_dict["e_train"], data_dict["t_train"],
                        data_dict["risk_set_valid"], data_dict["x_valid"], data_dict["e_valid"], data_dict["t_valid"], 
                          optimizer, n_epochs,
                          data_dict["x_test"],data_dict["e_test"],data_dict["t_test"],data_dict["risk_set_test"])   
    return metrics
    


# In[ ]:




# In[10]:

print("Risk sets calculated")


# In[11]:

dataset_dict={1:"WHAS",2:"GBSG",3:"METABRIC",4:"SUPPORT"}


# In[ ]:

def final_experiments():
    for i in range(2,5):
        x_train,train,x_valid,valid,x_test,test,name=get_dataset(i)
        print("Running for", name ,"dataset")
        
        x_train,e_train,t_train,x_valid,e_valid,t_valid,x_test,e_test,t_test = scale_data_to_torch(x_train,train,x_valid,valid,x_test,test)
        print("Dataset loaded and scaled")
        
        risk_set,risk_set_valid,risk_set_test=compute_risk_set(t_train,t_valid,t_test)
        print("Risk set computed")
        
        data_dict={"x_train":x_train,"e_train":e_train,"t_train":t_train,
                   "x_valid":x_valid,"e_valid":e_valid,"t_valid":t_valid,
                   "x_test":x_test,"e_test":e_test,"t_test":t_test,
                   "risk_set":risk_set,"risk_set_valid":risk_set_valid,"risk_set_test":risk_set_test
                  }
        
        n_in = x_train.shape[1]
        linear_models=[2,5,10,12]
        learning_rates=[1e-4,1e-3]
        layer_sizes = [[n_in],[n_in,n_in],[n_in,n_in,n_in],[n_in,20,15]]
        data=[data_dict]
        hyperparams = [(linear_model, learning_rate, layer_size, seed, d) for layer_size in layer_sizes for learning_rate in learning_rates 
               for linear_model in linear_models for seed in range(3) for d in data]
        print("Hyperparams initialized")

        p = Pool(32)
        print("Pool created")
        output = p.map(run_experiment, hyperparams)
        p.close()
        p.join()
#         a,b,c,d,e=hyperparams[0]
#         output=run_experiment(a,b,c,d,e)
        print("Models trained. Writing to file")
        filename=name+"_results"
        f = open(filename, "wb")
        pkl.dump(output, f)
        f.flush()
        f.close()
        print(name,"done")
        print("")
        


# In[ ]:

final_experiments()


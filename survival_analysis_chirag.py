import numpy
import time
import torch

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from torch import nn 
import pandas as pd
import numpy as np



def negative_log_likelihood(risk, E):
    hazard_ratio = torch.exp(risk)
    # print("hazard_ratio: ", hazard_ratio)
    log_risk = torch.log(torch.cumsum(hazard_ratio.flatten(), dim=0))
    # print("log_risk: ", log_risk)
    uncensored_likelihood = torch.transpose(risk, 0, 1) - log_risk
    censored_likelihood = uncensored_likelihood * E
    num_observed_events = torch.sum(E)
    # print("observed_events: ", num_observed_events)
    # print(censored_likelihood)
    neg_likelihood = -torch.sum(censored_likelihood)# / num_observed_events
    print("neg_likelihood: ", neg_likelihood)
    return neg_likelihood


def get_concordance_index(x, t, e, **kwargs):
    x = x.detach().cpu().numpy()
    t = t.detach().cpu().numpy()
    e = e.detach().cpu().numpy()
    computed_hazard = np.exp(x)

    return concordance_index(t,-1*computed_hazard,e)

def valid_fn(model,valid_dataloader,device,optimizer):

    optimizer.zero_grad()
    # print("x: ", x)
    outputs = model(x)

    loss = negative_log_likelihood(outputs, e)

    valid_ci = get_concordance_index(outputs,t,e)
    model.train()
    return valid_ci,loss



def train(model, train_dataloader, device, optimizer, valid_dataloader, n_epochs, standardize):
    offset = None
    scale = None
    patience = 2000
    improvement_threshold = 0.99999
    patience_increase = 2
    validation_frequency = 5

    # Initialize Metrics
    best_validation_loss = numpy.inf
    best_params = None
    best_params_idx = -1
    c_index = []
    valid_c_index = []

    start = time.time()
    for epoch in range(n_epochs):

        x = x.to(device)
        e = e.float().to(device)

        optimizer.zero_grad()
        # print("x: ", x)
        outputs = model(x)

        loss = negative_log_likelihood(outputs, e)
        loss.backward()
        optimizer.step()


        print loss.cpu().data.numpy()
        
        ci_train = get_concordance_index(outputs, t, e)
        c_index.append(ci_train)
        torch.cuda.empty_cache()

        del x
        del e
        del loss
            
        print(avg_loss)

        
        print('Finished Training with %d iterations in %0.2fs' % (epoch + 1, time.time() - start))
    
    metrics = {}
    metrics['best_valid_loss'] = best_validation_loss
    metrics['best_params_idx'] = best_params_idx
    metrics['c-index'] = c_index
    #metrics['valid_c-index'] = valid_c_index
    return metrics


def predict_risk(x):
    """
    Calculates the predicted risk for an array of observations.

    Parameters:
        x: (n,d) numpy array of observations.

    Returns:
        risks: (n) array of predicted risks
    """
    # risk_fxn = theano.function(
    #     inputs = [self.X],
    #     outputs = self.risk(deterministic= True),
    #     name = 'predicted risk'
    # )
    # return risk_fxn(x)
    pass

def recommend_treatment(x, trt_i, trt_j, trt_idx = -1):
    """
    Computes recommendation function rec_ij(x) for two treatments i and j.
        rec_ij(x) is the log of the hazards ratio of x in treatment i vs.
        treatment j.

    .. math::

        rec_{ij}(x) = log(e^h_i(x) / e^h_j(x)) = h_i(x) - h_j(x)

    Parameters:
        x: (n, d) numpy array of observations
        trt_i: treatment i value
        trt_j: treatment j value
        trt_idx: the index of x representing the treatment group column

    Returns:
        rec_ij: recommendation
    """
    # Copy x to prevent overwritting data
    x_trt = numpy.copy(x)

    # Calculate risk of observations treatment i
    x_trt[:,trt_idx] = trt_i
    h_i = predict_risk(x_trt)
    # Risk of observations in treatment j
    x_trt[:,trt_idx] = trt_j;
    h_j = predict_risk(x_trt)

    rec_ij = h_i - h_j
    return rec_ij

def plot_risk_surface(data, i = 0, j = 1,
    figsize = (6,4), x_lims = None, y_lims = None, c_lims = None):
    """
    Plots the predicted risk surface of the network with respect to two
    observed covarites i and j.

    Parameters:
        data: (n,d) numpy array of observations of which to predict risk.
        i: index of data to plot as axis 1
        j: index of data to plot as axis 2
        figsize: size of figure for matplotlib
        x_lims: Optional. If provided, override default x_lims (min(x_i), max(x_i))
        y_lims: Optional. If provided, override default y_lims (min(x_j), max(x_j))
        c_lims: Optional. If provided, override default color limits.

    Returns:
        fig: matplotlib figure object.
    """
    fig = plt.figure(figsize=figsize)
    X = data[:,i]
    Y = data[:,j]
    Z = predict_risk(data)

    if not x_lims is None:
        x_lims = [np.round(np.min(X)), np.round(np.max(X))]
    if not y_lims is None:
        y_lims = [np.round(np.min(Y)), np.round(np.max(Y))]
    if not c_lims is None:
        c_lims = [np.round(np.min(Z)), np.round(np.max(Z))]

    ax = plt.scatter(X,Y, c = Z, edgecolors = 'none', marker = '.')
    ax.set_clim(*c_lims)
    plt.colorbar()
    plt.xlim(*x_lims)
    plt.ylim(*y_lims)
    plt.xlabel('$x_{%d}$' % i, fontsize=18)
    plt.ylabel('$x_{%d}$' % j, fontsize=18)

    return fig

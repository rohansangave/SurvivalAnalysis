import numpy
import time
import torch

from lifelines.utils import concordance_index
from torch import nn 



def negative_log_likelihood(risk, E):
    hazard_ratio = torch.exp(risk)
    # print("hazard_ratio: ", hazard_ratio)
    log_risk = torch.log(torch.cumsum(hazard_ratio.flatten(), dim=0))
    # print("log_risk: ", log_risk)
    uncensored_likelihood = torch.transpose(risk, 0, 1) - log_risk
    censored_likelihood = uncensored_likelihood * E
    num_observed_events = torch.sum(E)
    # print("observed_events: ", num_observed_events)
    neg_likelihood = -torch.sum(censored_likelihood) # / num_observed_events
    print("neg_likelihood: ", neg_likelihood)
    return neg_likelihood


def get_concordance_index(x, t, e, **kwargs):
    """
    Taken from the lifelines.utils package. Docstring is provided below.

    Parameters:
        x: (n, d) numpy array of observations.
        t: (n) numpy array representing observed time events.
        e: (n) numpy array representing time indicators.

    Returns:
        concordance_index: calcualted using lifelines.utils.concordance_index

    lifelines.utils.concordance index docstring:

    Calculates the concordance index (C-index) between two series
    of event times. The first is the real survival times from
    the experimental data, and the other is the predicted survival
    times from a model of some kind.

    The concordance index is a value between 0 and 1 where,
    0.5 is the expected result from random predictions,
    1.0 is perfect concordance and,
    0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)

    Score is usually 0.6-0.7 for survival models.

    See:
    Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
    developing models, evaluating assumptions and adequacy, and measuring and
    reducing errors. Statistics in Medicine 1996;15(4):361-87.
    """
    # compute_hazards = theano.function(
    #     inputs = [self.X],
    #     outputs = -self.partial_hazard
    # )
    # partial_hazards = compute_hazards(x)

    # return concordance_index(t,
    #     partial_hazards,
    #     e)
    pass

# @TODO: implement for varios instances of datasets
def prepare_data(dataset, standardize=False, offset=None, scale=None):
    if isinstance(dataset, dict):
        x, e, t = dataset['x'], dataset['e'], dataset['t']

    if standardize:
        x = (x - offset) / scale

    # Sort Training Data for Accurate Likelihood
    sort_idx = numpy.argsort(t)[::-1]
    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    return (torch.from_numpy(x), torch.from_numpy(e), torch.from_numpy(t))

def train(model, train_dataloader, device, optimizer, scheduler, valid_dataloader= None, n_epochs = 500, standardize=True):
    """
    Trains a DeepSurv network on the provided training data and evalutes
        it on the validation data.

    Parameters:
        train_data: dictionary with the following keys:
            'x' : (n,d) array of observations (dtype = float32).
            't' : (n) array of observed time events (dtype = float32).
            'e' : (n) array of observed time indicators (dtype = int32).
        valid_data: optional. A dictionary with the following keys:
            'x' : (n,d) array of observations.
            't' : (n) array of observed time events.
            'e' : (n) array of observed time indicators.
        standardize: True or False. Set the offset and scale of
            standardization layey to the mean and standard deviation of the
            training data.
        n_epochs: integer for the maximum number of epochs the network will
            train for.
        validation_frequency: how often the network computes the validation
            metrics. Decreasing validation_frequency increases training speed.
        patience: minimum number of epochs to train for. Once patience is
            reached, looks at validation improvement to increase patience or
            early stop.
        improvement_threshold: percentage of improvement needed to increase
            patience.
        patience_increase: multiplier to patience if threshold is reached.
        logger: None or DeepSurvLogger.
        update_fn: lasagne update function for training.
            Default: lasagne.updates.nesterov_momentum
        **kwargs: additional parameters to provide _get_train_valid_fn.
            Parameters used to provide configurations to update_fn.

    Returns:
        metrics: a dictionary of training metrics that include:
            'train': a list of loss values for each training epoch
            'train_ci': a list of C-indices for each training epoch
            'best_params': a list of numpy arrays containing the parameters
                when the network had the best validation loss
            'best_params_idx': the epoch at which best_params was found
        If valid_data is provided, the metrics also contain:
            'valid': a list of validation loss values for each validation frequency
            'valid_ci': a list of validation C-indiices for each validation frequency
            'best_validation_loss': the best validation loss found during training
            'best_valid_ci': the max validation C-index found during training
    """

    offset = None
    scale = None
    patience = 2000
    improvement_threshold = 0.99999
    patience_increase = 2
    validation_frequency = 5

    # Set Standardization layer offset and scale to training data mean and std
    # if standardize:
    #     offset = train_data['x'].mean(axis = 0)
    #     scale = train_data['x'].std(axis = 0)

    # x_train, e_train, t_train = prepare_data(train_data, standardize, offset, scale)

    # if valid_data:
    #     x_valid, e_valid, t_valid = prepare_data(valid_data, standardize, offset, scale)

    # Initialize Metrics
    best_validation_loss = numpy.inf
    best_params = None
    best_params_idx = -1

    start = time.time()
    for epoch in range(n_epochs):
        scheduler.step()
        avg_loss = 0.0
        i = 0
        for batch_idx, (x, e, t) in enumerate(train_dataloader):
            x = x.to(device)
            e = e.float().to(device)

            optimizer.zero_grad()
            # print("x: ", x)
            outputs = model(x)

            loss = negative_log_likelihood(outputs, e)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

            i += 1
            if i % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, i+1, avg_loss/50))
                avg_loss = 0.0    
            
            ci_train = get_concordance_index(x, t, e)

            torch.cuda.empty_cache()
            del x
            del e
            del loss

        # not completed yet
        if valid_dataloader and (epoch % validation_frequency == 0):
            validation_loss = valid_fn(x_valid, e_valid)
            print('valid_loss: ',validation_loss)

            ci_valid = get_concordance_index(
                x_valid,
                t_valid,
                e_valid
            )

            if validation_loss < best_validation_loss:
                # improve patience if loss improves enough
                if validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, epoch * patience_increase)

                model_filename = "best_model_epoch_" + str(epoch) + ".pt"
                torch.save(model.state_dict(), model_filename)
                best_params_idx = epoch
                best_validation_loss = validation_loss

        if patience <= epoch:
            break    
        
        print('Finished Training with %d iterations in %0.2fs' % (epoch + 1, time.time() - start))
    
    metrics = {}
    metrics['best_valid_loss'] = best_validation_loss
    metrics['best_params_idx'] = best_params_idx

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

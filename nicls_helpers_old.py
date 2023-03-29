import warnings
warnings.simplefilter('ignore')
import sys
sys.path.append('/home1/jrudoler/src/')
import cmlreaders as cml
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import seaborn as sns
import pandas as pd
from pd_to_pb import pandas_to_pybeh as pb
import scipy as scp
from classifier_io import ClassifierModel
from sklearn.linear_model import LogisticRegression
from sklearn import __version__ as sklearn_version
import pickle
from ptsa.data.filters import MorletWaveletFilter, ButterworthFilter
from ptsa.data.timeseries import TimeSeries
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from numpy.random import shuffle

def norm_sess_feats(feats, n_lists):
    mu = feats.query(event=f"trial<{n_lists}").mean('event')
    std = feats.query(event=f"trial<{n_lists}").std('event', ddof=1)
    norm_pows =  (feats - mu) / std
    return norm_pows

def n_list_normalization(n_lists, subject_list, c_list = np.logspace(np.log10(2e-5), np.log10(1), 9), path='/scratch/nicls_intermediate/read_only/encoding_powers/'):
    result_dict = {}
    for subject in subject_list:
        ts = TimeSeries.from_hdf(path+subject+"_feats.h5", engine="netcdf4")
        evs = ts.indexes['event'].to_frame(index=False)
        ts = ts.groupby('session').apply(norm_sess_feats, n_lists=n_lists)
        scores_mat = computeCV(ts.data, evs, c_list)
        best_c = c_list[scores_mat.mean(0).argmax()]
        model = LogisticRegression(penalty='l2', C=best_c, class_weight='balanced', solver='liblinear')
        prob = perform_normed_loso_cross_validation(model, events=evs, powers=ts, n_lists=n_lists)
        fp, tp, _ = roc_curve(evs.recalled.values, prob)
        auc = roc_auc_score(evs.recalled.values, prob)
        result_dict[subject] = (fp, tp, auc)

    with open(f"{n_lists}_list_normalization.pkl", "wb") as f:
        pickle.dump(result_dict, f)

        
def train_read_only_class(subject, c_list, path='/scratch/nicls_intermediate/read_only/encoding_powers/'):
    ts = TimeSeries.from_hdf(path+subject+"_feats.h5", engine="netcdf4")
    evs = ts.indexes['event'].to_frame(index=False)
    ts = ts.groupby('session').reduce(scp.stats.zscore, dim='event', ddof=1)
    scores_mat = computeCV(ts.data, evs, c_list)
    best_c = c_list[scores_mat.mean(0).argmax()]
    model = LogisticRegression(penalty='l2', C=best_c, class_weight='balanced', solver='liblinear')
    prob = perform_loso_cross_validation(model, events=evs, powers=ts.data)
    fp, tp, _ = roc_curve(evs.recalled.values, prob)
    auc = roc_auc_score(evs.recalled.values, prob)
    auc_results = [roc_auc_score(np.random.choice(evs.recalled.values,
                                                  size=len(evs),
                                                 replace=False),
                                 prob)
                   for i in range(1000)]
    return (fp, tp, auc, auc_results, model)


def perform_normed_loso_cross_validation(classifier, powers, events, n_lists, **kwargs):
    """ Perform single iteration of leave-one-session-out cross validation
    Parameters
    ----------
    classifier:
        sklearn model object, usually logistic regression classifier
    powers: np.ndarray
        power matrix
    events : np.recarray
    kwargs: dict
        Optional keyword arguments. These are passed to get_sample_weights.
        See that function for more details.
    Returns
    -------
    probs: np.array
        Predicted probabilities for encoding events across all sessions
    """
    classifier_copy = deepcopy(classifier)
    sessions = np.unique(events.session)
    recalls = events.recalled.values.astype(int)

    # Predicted probabilities should be assessed only on encoding words
    probs = np.empty_like(recalls, dtype=float)

    for sess_idx, sess in enumerate(sessions):
        # training data
        insample_mask = (powers.session != sess)
        insample_pow_mat = powers.sel({'event':insample_mask})
        insample_pow_mat = insample_pow_mat.groupby('session').reduce(scp.stats.zscore, dim='event', ddof=1).data
        # normalize with zcore
        insample_recalls = recalls[insample_mask]
        classifier_copy.fit(insample_pow_mat, insample_recalls)

        # testing data 
        outsample_mask = ~insample_mask #& encoding_mask
        outsample_pow_mat = powers.sel(event=outsample_mask)
        outsample_pow_mat = outsample_pow_mat.groupby('session').apply(norm_sess_feats, n_lists=n_lists).data

        outsample_probs = classifier_copy.predict_proba(outsample_pow_mat)[:, 1]

        probs[outsample_mask] = outsample_probs

    return probs

def perform_loso_cross_validation(classifier, powers, events, **kwargs):
    """ Perform single iteration of leave-one-session-out cross validation
    Parameters
    ----------
    classifier:
        sklearn model object, usually logistic regression classifier
    powers: np.ndarray
        power matrix
    events : np.recarray
    kwargs: dict
        Optional keyword arguments. These are passed to get_sample_weights.
        See that function for more details.
    Returns
    -------
    probs: np.array
        Predicted probabilities for encoding events across all sessions
    """
    classifier_copy = deepcopy(classifier)
    sessions = np.unique(events.session)
    recalls = events.recalled.values.astype(int)

    # Predicted probabilities should be assessed only on encoding words
    probs = np.empty_like(recalls, dtype=float)

    for sess_idx, sess in enumerate(sessions):
        # training data
        insample_mask = (events.session != sess)
        insample_pow_mat = powers[insample_mask]
        insample_recalls = recalls[insample_mask]
        classifier_copy.fit(insample_pow_mat, insample_recalls)

        # testing data 
        outsample_mask = ~insample_mask #& encoding_mask
        outsample_pow_mat = powers[outsample_mask]

        outsample_probs = classifier_copy.predict_proba(outsample_pow_mat)[:, 1]

        probs[outsample_mask] = outsample_probs

    return probs

def permuted_loso_cross_validation(classifier, powers, events, n_permutations, **kwargs):
    """ Perform permuted leave one session out cross validation

    Parameters
    ----------
    classifier:
        sklearn model object, usually logistic regression classifier
    powers: np.ndarray
        power matrix
    events : recarray
    n_permutations: int
        number of permutation trials
    kwargs: dict
        Optional keyword arguments. These are passed to get_sample_weights.
        See that function for more details.

    Returns
    -------
    AUCs: list
        List of AUCs from performing leave-one-list-out cross validation
        n_permutations times where the AUCs are based on encoding events only

    """
    recalls = events.recalled
    sessions = np.unique(events.session)

    permuted_recalls = recalls.copy()
    auc_results = np.empty(shape=n_permutations, dtype=float)

    for i in range(n_permutations):
        # Shuffle recall outcomes within session
        for session in sessions:
            in_session_mask = (events.session == session)
            session_permuted_recalls = permuted_recalls[in_session_mask]
            session_permuted_recalls = session_permuted_recalls.sample(frac=1).values
            permuted_recalls[in_session_mask] = session_permuted_recalls

        # The probabilities returned here will be only for encoding events
        probs = perform_loso_cross_validation(classifier, powers, events, **kwargs)

        # Evaluation should happen only on encoding events
#         encoding_recalls = permuted_recalls[encoding_mask]
        auc_results[i] = roc_auc_score(permuted_recalls, probs)

    return auc_results

def computeCV(full_pows, full_evs, c_list):
    # Selecting overall parameter using nested CV, with LOLO cross validation to get scores for every
    # parameter and session, then averaging across sessions and taking the paramter which yeilds the highest
    # average AUC
    all_scores = []
    for sess in full_evs.session.unique():
        out_mask = full_evs.session == sess
        in_mask = ~out_mask
        score_list = []
        for c in c_list:
            model = LogisticRegression(penalty='l2', C=c, solver='liblinear')
            model.fit(full_pows[in_mask], full_evs[in_mask].recalled.astype(int))
            probs = model.predict_proba(full_pows[out_mask])[:, 1]
            score_list.append(roc_auc_score(full_evs[out_mask].recalled.astype(int), probs))
        all_scores.append(score_list)
    # return scores matrix shaped sessions x hyperparameter
    scores_mat = np.stack(all_scores)
    return scores_mat

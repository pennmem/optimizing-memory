import warnings
warnings.simplefilter('ignore')
import sys
sys.path.append('/home1/jrudoler/src/')
import cmlreaders as cml
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import seaborn as sns
import pandas as pd
import scipy as scp
import pickle
from ptsa.data.filters import MorletWaveletFilter, ButterworthFilter
from ptsa.data.timeseries import TimeSeries
from copy import deepcopy
from numpy.random import shuffle
random_seed = 56
np.random.seed(random_seed)
random.seed(random_seed)

# ML stuff
from classifier_io import ClassifierModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn import __version__ as sklearn_version
from sklearn.utils import parallel_backend
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.multitest import fdrcorrection
from joblib.parallel import Parallel, delayed

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

def NestedLOGO(data, estimator, param_grid, inner_cv_group='session', outer_cv_group='session', client=None, verbose=True):
    outer_cv = LeaveOneGroupOut()
    inner_cv = LeaveOneGroupOut()
    results = {"subject": data['subject'].values[0], "holdout_session":[], "penalty":[], "AUC":[], "null":[],
               "pval":[], "fp":[], "tp":[], "y_true":[], "y_pred":[], "model":[], "activations":[]}
    ## outer cv over sessions, holding one out
    models = []
    for train_idx, test_idx in outer_cv.split(X=data.data,
                                              y=data.recalled.values,
                                              groups=data[outer_cv_group].values):
        ## inner cv fold over remaining sessions to select penalty
        inner_model = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv.split(X=data.data[train_idx],
                              y=data.recalled.values[train_idx],
                              groups=data[inner_cv_group].values[train_idx]),
            scoring='roc_auc',
            n_jobs=1, verbose=verbose
        )
        inner_model.fit(X=data.data[train_idx], y=data.recalled.values[train_idx])
        ## automatically uses best fit from inner cv
        predictions = inner_model.predict_proba(X=data.data[test_idx])[:, 1]
        if np.mean(data.recalled.values[test_idx]) in [0., 1.]:
            auc = np.nan
            auc_null = None
            p = np.nan
            fp = None
            tp = None
        else:
            auc, auc_null, p = post_hoc_permutation(y_true=data.recalled.values[test_idx], 
                                                    y_score=predictions)
            fp, tp, _ = roc_curve(data.recalled.values[test_idx], predictions)
        # compute feature activations
        activation = compute_activation(data.data[train_idx], inner_model.best_estimator_.coef_.flatten())
        activation = activation.reshape((8, 128))
        ## update results
        results["holdout_session"].append(data.session[test_idx].values[0])
        results["y_true"].append(data.recalled.values[test_idx])
        results["y_pred"].append(predictions)
        results["AUC"].append(auc)
        results["null"].append(auc_null)
        results["penalty"].append(inner_model.best_params_['C'])
        results["pval"].append(p)
        results["fp"].append(fp)
        results["tp"].append(tp)
        results["model"].append(inner_model.best_estimator_)
        results["activations"].append(activation)
    return pd.DataFrame(results)

def compute_activation(X, weights):
    '''A = cov(X) * W / cov(y_hat)'''
    # where X = train_data.values
    
    activations = np.cov(X.T).dot(weights) / np.cov(X.dot(weights))

    # activations shape: N features array
    return activations

def load_and_fit_data_encoding(path, group):
    estimator = LogisticRegression(penalty='l2', class_weight='balanced', solver='lbfgs', max_iter=1000)
    param_grid = {"C":np.power(10., -np.arange(6))}
    encoding_data = TimeSeries.from_hdf(path)
    phase = np.where(encoding_data['trial']<5, 0, 1)
    encoding_data = encoding_data.assign_coords(
        subsession=("event", encoding_data["session"].values*2+phase),
        sess_trial=("event", encoding_data["session"].values*10+encoding_data["trial"].values)
    )
    encoding_data = encoding_data.query({"event":"subsession>0 & trial>=0"})
    results = NestedLOGO(encoding_data, estimator, param_grid,
                         inner_cv_group=group, outer_cv_group=group)
    return results


def bordered_imshow(X, mask, line_kwargs={"linewidth":4, "color":'k'}, **kwargs):
    shape = X.shape
    im = plt.imshow(X, extent=[0, shape[0], 0, shape[1]], origin='lower', **kwargs)
    ax = plt.gca()
    extent = im.get_extent()
    masked_cells = [(i, j) for i, j in zip(*np.nonzero(mask))]
    for cell in masked_cells:
        # left
        if (cell[0], cell[1]-1) not in masked_cells:
            line = plt.Line2D([extent[0]+(extent[1]/shape[1])*cell[1], extent[0]+(extent[1]/shape[1])*cell[1]],
                              [extent[2]+(extent[3]/shape[0])*cell[0], extent[2]+(extent[1]/shape[0])*cell[0]+1],
                              **line_kwargs)
            ax.add_patch(line)
        # right
        if (cell[0], cell[1]+1) not in masked_cells:
            line = plt.Line2D([extent[0]+(extent[1]/shape[1])*cell[1]+1, extent[0]+(extent[1]/shape[1])*cell[1]+1],
                              [extent[2]+(extent[3]/shape[0])*cell[0], extent[2]+(extent[1]/shape[0])*cell[0]+1],
                              **line_kwargs)
            ax.add_patch(line)
        # above
        if (cell[0]+1, cell[1]) not in masked_cells:
            line = plt.Line2D([extent[0]+(extent[1]/shape[1])*cell[1], extent[0]+(extent[1]/shape[1])*cell[1]+1],
                              [extent[2]+(extent[3]/shape[0])*cell[0]+1, extent[2]+(extent[1]/shape[0])*cell[0]+1],
                              **line_kwargs)
            ax.add_patch(line)
        # below
        if (cell[0]-1, cell[1]) not in masked_cells:
            line = plt.Line2D([extent[0]+(extent[1]/shape[1])*cell[1], extent[0]+(extent[1]/shape[1])*cell[1]+1],
                              [extent[2]+(extent[3]/shape[0])*cell[0], extent[2]+(extent[1]/shape[0])*cell[0]],
                              **line_kwargs)
            ax.add_patch(line)
    return im, ax

def post_hoc_permutation(y_true, y_score, n_permutations=4999, score_function=roc_auc_score, seed=None, n_jobs=None, backend="threading", verbose=False): 
    if seed:
        np.random.seed(seed)
    score = score_function(y_true, y_score)
    permutation_scores = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(score_function)(
            np.random.choice(y_true, len(y_true), replace=False),
            y_score
        )
        for _ in range(n_permutations)
    )
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= score) + 1.) / (n_permutations + 1.)
    return score, permutation_scores, pvalue 

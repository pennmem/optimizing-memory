#!/usr/bin/env python

import sys

import sys
import cmlreaders as cml
import numpy as np
import seaborn as sns
import pandas as pd
from ptsa.data.filters import MorletWaveletFilter, ButterworthFilter, MonopolarToBipolarMapper
from ptsa.data.timeseries import TimeSeries
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import xarray
from ptsa.data.concat import concat
import cmldask.CMLDask as da

def compute_raw_event_features(events, subject, settings_path, save_path='/scratch/jrudoler/NICLS/report_data/closed_loop/encoding_powers/'):
    """
    Compute log-transformed powers, averaged over time and stacked as (frequency, channel) to create features
    These can later be normalized along the event axis.
    """
    settings = da.Settings.Load(settings_path)
    feats = []
    for session, evs in events.groupby("session"):
        r = cml.CMLReader(subject=subject, experiment=settings.experiment, session=session)
        eeg = r.load_eeg(evs,
                         rel_start=settings.rel_start - settings.buffer_time,
                         rel_stop=settings.rel_stop + settings.buffer_time,
                         clean=settings.clean
                        ).to_ptsa()
        # select relevant channels
        if settings.reference == 'average':
            eeg = eeg[:, :128]
            if eeg.channel[0].str.startswith('E') and not settings.clean: # EGI system
                eeg.drop_sel({'channel':['E8', 'E25', 'E126', 'E127']})
            eeg -= eeg.mean('channel')
        elif settings.reference == 'bipolar':
            bipolar_pairs = np.loadtxt("/home1/jrudoler/biosemi_cap_bipolar_pairs.txt", dtype=str)
            mapper = MonopolarToBipolarMapper(bipolar_pairs, channels_dim='channel')
            eeg = eeg.filter_with(mapper)
            eeg = eeg.assign_coords({"channel":np.array(["-".join(pair) for pair in eeg.channel.values])})
        else:
            raise ValueError("reference setting unknown")
        # filter out line noise at 60 and 120Hz
        eeg = ButterworthFilter(filt_type='stop', freq_range=[58, 62], order=4).filter(eeg)
        eeg = ButterworthFilter(filt_type='stop', freq_range=[118, 122], order=4).filter(eeg)
        # highpass filter to account for drift
        eeg = ButterworthFilter(filt_type='highpass', freq_range=1).filter(eeg)
        pows = MorletWaveletFilter(settings.freqs,
                                   width=settings.width,
                                   output='power',
                                   cpus=25).filter(eeg)
        del eeg
        pows = pows.remove_buffer(settings.buffer_time / 1000) + np.finfo(float).eps/2.
        pows = pows.reduce(np.log10)
        # swap order of events and frequencies --> result is events x frequencies x channels x time
        # next, average over time
        pows = pows.transpose('event', 'frequency', 'channel', 'time').mean('time')
        # reshape as events x features
        pows = pows.stack(features=("frequency", "channel"))
        feats.append(pows)
        del pows
    feats = concat(feats, dim='event')
    feats = feats.assign_attrs(settings.__dict__)
    if settings.save:
        feats.to_hdf(save_path+f'{subject}_raw_feats.h5')
    return feats

if __name__=="__main__":
    #compute_features(sys.argv[1], experiment="NiclsCourierReadOnly", save=True, save_path='/scratch/nicls_intermediate/read_only/encoding_powers/')
    compute_scalp_features(sys.argv[1], experiment="NiclsCourierClosedLoop", save=True, save_path='/scratch/nicls_intermediate/closed_loop/encoding_powers/')
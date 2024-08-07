#!/usr/bin/env python

import sys
import os
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from ptsa.data.filters import (
    MorletWaveletFilter,
    ButterworthFilter,
    MonopolarToBipolarMapper,
)
from ptsa.data.timeseries import TimeSeries
from scipy.stats import zscore
import xarray
from ptsa.data.concat import concat
import pickle
from scipy.stats import zscore


def mne_to_ptsa(ep):
    """Create an PTSA TimeSeries (essentially xarray) version of MNE epoch data"""
    assert ep.metadata is not None, "Please define mne.Epoch.metadata"
    ts = TimeSeries(
        ep.get_data(copy=True),
        dims=("event", "channel", "time"),
        coords={
            "event": pd.MultiIndex.from_frame(ep.metadata),
            "channel": ep.info["ch_names"],
            "time": ep.times,
            "samplerate": ep.info["sfreq"],
        },
        name="data",
    )
    ts = ts.reset_index("event")
    return ts


def compute_scalp_features(
    subject,
    settings_path,
    normalize=False,
    bids_root="~/data/nicls_bids/",
    save_path="~/data/features/",
):
    """
    Compute log-transformed powers, averaged over time and stacked as (frequency, channel) to create features
    Normaize within each session.
    """
    # expand paths
    bids_root = os.path.expanduser(bids_root)
    settings_path = os.path.expanduser(settings_path)
    save_path = os.path.expanduser(save_path)

    # load settings
    with open(settings_path, "rb") as f:
        settings = pickle.load(f)
    # find all sessions for subject
    bids_paths = find_matching_paths(
        bids_root,
        subjects=subject,
        tasks="NiclsCourierReadOnly",
        datatypes="eeg",
        extensions=".bdf",
    )
    # read in data and compute features one session at a time
    feats = []
    for i, bids_path in enumerate(bids_paths):
        print(f"Reading session {i} data")
        # intialize data reader, load words events and buffered eeg epochs
        events_path = bids_path.copy().update(suffix="events", extension=".tsv")
        event_df = pd.read_csv(events_path, sep="\t")
        raw = read_raw_bids(
            bids_path, extra_params={"infer_types": True}, verbose=False
        )
        mne_events, mne_event_id = mne.events_from_annotations(raw, verbose=False)
        eeg = mne.Epochs(
            raw,
            mne_events,
            event_id={"WORD": mne_event_id["WORD"]},
            event_repeated="drop",
            tmin=(settings["rel_start"] - settings["buffer_time"]) / 1000.0,
            tmax=(settings["rel_stop"] + settings["buffer_time"]) / 1000.0,
            baseline=None,
            preload=True,
        )
        eeg.metadata = event_df.query("trial_type=='WORD'").reset_index(drop=True)
        eeg = mne_to_ptsa(eeg)
        # select relevant channels
        if settings["reference"] == "average":
            eeg = eeg[:, :128]
            if (
                eeg.channel[0].str.startswith("E") and not settings["clean"]
            ):  # EGI system
                eeg.drop_sel({"channel": ["E8", "E25", "E126", "E127"]})
            eeg -= eeg.mean("channel")
        elif settings["reference"] == "bipolar":
            bipolar_pairs = np.loadtxt(
                "/home1/jrudoler/biosemi_cap_bipolar_pairs.txt", dtype=str
            )
            mapper = MonopolarToBipolarMapper(bipolar_pairs, channels_dim="channel")
            eeg = eeg.filter_with(mapper)
            eeg = eeg.assign_coords(
                {"channel": np.array(["-".join(pair) for pair in eeg.channel.values])}
            )
        else:
            raise ValueError("reference setting unknown")
        # filter out line noise at 60 and 120Hz
        eeg = ButterworthFilter(filt_type="stop", freq_range=[58, 62], order=4).filter(
            eeg
        )
        eeg = ButterworthFilter(
            filt_type="stop", freq_range=[118, 122], order=4
        ).filter(eeg)
        # highpass filter to account for drift
        eeg = ButterworthFilter(filt_type="highpass", freq_range=1).filter(eeg)
        pows = MorletWaveletFilter(
            settings["freqs"], width=settings["width"], output="power", cpus=25
        ).filter(eeg)
        del eeg
        pows = (
            pows.remove_buffer(settings["buffer_time"] / 1000)
            + np.finfo(float).eps / 2.0
        )
        pows = pows.reduce(np.log10)
        # swap order of events and frequencies --> result is events x frequencies x channels x time
        # next, average over time
        pows = pows.transpose("event", "frequency", "channel", "time").mean("time")
        # reshape as events x features
        pows = pows.stack(features=("frequency", "channel")).reset_index("features")
        # normalize along event axis
        if normalize:
            pows = pows.reduce(func=zscore, dim="event", keep_attrs=True, ddof=1)
        feats.append(pows)
        del pows
    feats = concat(feats, dim="event")
    settings.update({"normalize": int(normalize)})
    feats = feats.assign_attrs(settings)
    suffix = "_feats.h5" if normalize else "_raw_feats.h5"
    if settings["save"]:
        os.path.makedirs(save_path, exist_ok=True)
        feats.to_hdf(os.path.join(save_path, f"{subject}_{suffix}"))
    return feats


if __name__ == "__main__":
    compute_scalp_features(
        sys.argv[1],
        experiment="NiclsCourierClosedLoop",
        normalize=True,
        save_path="/scratch/nicls_intermediate/closed_loop/encoding_powers/",
    )

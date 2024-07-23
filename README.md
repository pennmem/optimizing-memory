# Decoding EEG for Optimizing Naturalistic Memory

This repository contains code for reproducing the analyses in the manuscript ["Decoding EEG for Optimizing Naturalistic Memory"](https://doi.org/10.1016/j.jneumeth.2024.110220).

The raw electrophysiological data is available on [OpenNeuro](https://openneuro.org/datasets/ds004706). For code used in running the experiment, see [here](https://github.com/pennmem/NICLS) and [here](https://github.com/pennmem/courier).

A few analyses required integrating data from raw session logs, which are currently not included in the OpenNeuro dataset (but may be added at a future date). For now, the necessary data from the logs used in analyzing closed-loop sessions can be found in `data/processed_events_NiclsCourierClosedLoop`. 
Essentially this is a tabular dataset that matches events from the backend which contain information about the classifier to events from the behavioral task which contain information about behavior/gameplay.

For any questions, please create an issue or contact [jrudoler@wharton.upenn.edu](mailto:jrudoler.wharton.upenn.edu).

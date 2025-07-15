import os, sys
import numpy as np
import h5py
import json

from . import agglomerate

def getScoreFunc(scoreF):
    # aff50_his256
    config = {x[:3]: x[3:] for x in scoreF.split('_')}
    if 'aff' in config:
        if 'his' in config and config['his']!='0':
            return 'OneMinus<HistogramQuantileAffinity<RegionGraphType, %s, ScoreValue, %s>>' % (config['aff'],config['his'])
        else:
            return 'OneMinus<QuantileAffinity<RegionGraphType, '+config['aff']+', ScoreValue>>'
    elif 'max' in config:
            return 'OneMinus<MeanMaxKAffinity<RegionGraphType, '+config['max']+', ScoreValue>>'

def getRegionGraph(affs, fragments, rg_opt = 1, merge_function = None, discretize_queue=256, rebuild = True):
    for rg in agglomerate(
            affs.astype(np.float32),
            thresholds = [0.1],
            fragments = fragments.astype(np.uint64),
            scoring_function = getScoreFunc(merge_function),
            discretize_queue = discretize_queue,
            rg_opt = rg_opt,
            force_rebuild=rebuild):
        return rg

def waterzFromRegionGraph(rg_id, rg_score, thresholds, merge_function = None, discretize_queue=256, rebuild = True):
    for merge_history in agglomerate(
            rg_score.astype(np.float32),
            thresholds = thresholds,
            fragments = rg_id.astype(np.uint64),
            scoring_function = getScoreFunc(merge_function),
            discretize_queue = discretize_queue,
            rg_opt = 3,
            force_rebuild=rebuild):
        return merge_history

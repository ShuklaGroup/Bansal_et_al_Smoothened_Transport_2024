from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov import TransitionCountEstimator
from deeptime.plots import plot_implied_timescales
from deeptime.util.validation import implied_timescales
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm
import os

cutoffs = [50]
lags=[300] #np.arange(10,601,10)
clus = [350]#range(100, 1001, 50)#[100,200,400,600,800,1000],1200,1400,1600,1800]
for ct in tqdm(cutoffs):
    for k in tqdm(clus):
        models = []
        dtrajs = pickle.load(open(f'./clus_tica_dtrajs/clus_tica_dtrajs_combined_{k}_{ct}.pkl','rb'))
        for lag in tqdm(lags):
            count_model = TransitionCountEstimator(lag, 'sliding').fit_fetch(dtrajs)
            msm = MaximumLikelihoodMSM().fit_fetch(count_model.submodel_largest())
        weights = msm.compute_trajectory_weights(dtrajs)
pickle.dump(msm, open(f'./msm_combined_{k}_{ct}_{lag}.pkl','wb'))
pickle.dump(weights, open(f'./weights_msm_combined_{k}_{ct}_{lag}.pkl','wb'))

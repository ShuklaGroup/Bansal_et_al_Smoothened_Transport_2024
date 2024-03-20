from deeptime.markov.msm import BayesianMSM
from deeptime.markov import TransitionCountEstimator
from deeptime.plots import plot_implied_timescales
from deeptime.util.validation import implied_timescales
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm
import os

cutoffs = [50]#[95,80,65,50]
lags=np.arange(10,601,10)
clus = [350]#range(100, 1001, 50)#[100,200,400,600,800,1000],1200,1400,1600,1800]
for ct in tqdm(cutoffs):
    for k in tqdm(clus):
        models = []
        #clus_tica_dtrajs_combined_
        dtrajs = pickle.load(open(f'./clus_tica_dtrajs/clus_tica_dtrajs_combined_{k}_{ct}.pkl','rb'))
        for lag in tqdm(lags):
            count_model = TransitionCountEstimator(lag, 'effective').fit_fetch(dtrajs)
            msm = BayesianMSM(n_samples=50).fit_fetch(count_model)
            #pickle.dump(msm, open('./bmsm_400_80_300.pkl','wb'))
            #pickle.dump(msm,open(f'./msmobj_{k}_{ct}.pkl','wb'))
            #scores = msm.score(dtrajs,dim=10)
            #pickle.dump(scores, open(f'./VAMPscore/VAMP2score_{k}_{ct}_{lag}.pkl','wb'))
            models.append(msm)
        pickle.dump(models, open(f'./bmsm_models_for_its_combined_{k}_{ct}.pkl','wb'))
        its_data = implied_timescales(models)
        pickle.dump(its_data, open(f'./its_combined_{k}_{ct}.pkl','wb'))
        fig, ax = plt.subplots(1, 1)
        plot_implied_timescales(its_data, n_its=2, ax=ax)
        ax.set_yscale('log')
        ax.set_title(f'Implied timescales, {k}, {ct}')
        ax.set_xlabel('lag time (steps)')
        ax.set_ylabel('timescale (steps)')
        fig.savefig(f'./itsplots/bmsm_ITS_plot_combined_{k}_{ct}.jpg',dpi=300)

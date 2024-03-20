import pickle
from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM
from deeptime.markov import TransitionCountEstimator
from deeptime.plots.chapman_kolmogorov import plot_ck_test
import numpy as np
from tqdm import tqdm

bmsm = pickle.load(open('./bmsm_400_80_300.pkl','rb'))
#data = pickle.load(open(f'./its_{k}_{ct}.pkl','rb'))
#lags=np.arange(10,601,10)
#k=400
#ct=80
#models=[]
#dtraj = pickle.load(open(f'./clus_tica_dtrajs/clus_tica_dtrajs_{k}_{ct}.pkl','rb'))
#for lag in tqdm(lags[-5:]):
#    counts_estimator = TransitionCountEstimator(lagtime=lag, count_mode='effective')
#    models.append(BayesianMSM(n_samples=5).fit_fetch(counts_estimator.fit_fetch(dtraj).submodel_largest()))

models = pickle.load(open('./bmsm_models_for_its_400_80.pkl','rb'))
ck_test = bmsm.ck_test(models, n_metastable_sets=5)
grid = plot_ck_test(ck_test, legend=False)
grid.figure.savefig('./CKtest_TM56.png',dpi=300,transparent=True)

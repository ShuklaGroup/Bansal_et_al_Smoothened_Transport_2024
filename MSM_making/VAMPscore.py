import numpy as np
import pyemma
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import seaborn as sns
import os

#clus = range(100, 1001, 50)
clus = list(range(100, 501, 50))+list(range(600, 1001, 100))
cutoffs = [95,80,65,50]
for clu in tqdm(clus):
  for ct in tqdm(cutoffs):
    cluster_tica_dtrajs = pickle.load(open(f'./clus_tica_dtrajs/clus_tica_dtrajs_combined_{clu}_{ct}.pkl','rb'))
    if os.path.exists(f'./VAMPscore/VAMP2_combined_{clu}_{ct}_5.npy'):
      continue
    msm=pyemma.msm.estimate_markov_model(dtrajs=cluster_tica_dtrajs,lag=300,score_method='VAMP2',score_k=5)
    #pickle.dump(msm,open(f'msmobj_{clu}_{ct}_5.pkl','wb'))
    score=msm.score_cv(dtrajs=cluster_tica_dtrajs)
    np.save(f'./VAMPscore/VAMP2_combined_{clu}_{ct}_5.npy',score)

import numpy as np
import pickle
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov import TransitionCountEstimator
from random import sample
from tqdm import tqdm
import os
from pdb3 import lsext
from multiprocessing import Pool

k=400
ct=80

dtrajs = pickle.load(open(f'./clus_tica_dtrajs/clus_tica_dtrajs_{k}_{ct}.pkl','rb'))
print(len(dtrajs))
len_total_trajs = len(dtrajs)
len_btsp_trajs = int(0.8*len_total_trajs)

def make_btsp_indices(n_samples):
    for i in tqdm(range(n_samples)):
        indices = sample(range(len_total_trajs), len_btsp_trajs)
        if not os.path.exists(f'./bt/indices_56_{i+1}.pkl'):
            pickle.dump(list(indices), open(f'./bt/indices_56_{i+1}.pkl','wb'))


def make_btsp_msms(indices_path):
    lag=300
    i = indices_path.split('/')[-1].split('_')[-1].split('.')[0]
    indices = pickle.load(open(indices_path,'rb'))
    dtrajs_btsp = [dtrajs[j] for j in indices]
    count_model = TransitionCountEstimator(lag, 'sliding').fit_fetch(dtrajs_btsp)
    msm = MaximumLikelihoodMSM().fit_fetch(count_model.submodel_largest())
    weights = msm.compute_trajectory_weights(dtrajs_btsp)
    pickle.dump(msm, open(f'./bt/msm_btsp_56_{i}.pkl','wb'))
    pickle.dump(weights, open(f'./bt/msm_weights_btsp_56_{i}.pkl','wb'))

if __name__=="__main__":
    indices_paths = lsext('./bt/','indices_56')[0]
    with Pool(processes=4) as p:
        max_ = len(indices_paths)
        with tqdm(total=max_) as pbar:
            for _ in p.imap_unordered(make_btsp_msms, indices_paths):
                pbar.update()

    #make_btsp_indices(200)

#count_model = TransitionCountEstimator(lag, 'sliding').fit_fetch(dtrajs)
#msm = MaximumLikelihoodMSM().fit_fetch(count_model.submodel_largest())
#weights = msm.compute_trajectory_weights(dtrajs)

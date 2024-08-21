from pdb3 import lsext
from tqdm.notebook import tqdm
import deeptime
import numpy as np
from deeptime.decomposition import TICA
import pickle
import argparse
from deeptime.clustering import KMeans, MiniBatchKMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM

lagtime = 300

transformed_data = pickle.load(open('/home/pdb3/SMO/6XBL_CHL/Analysis/remaking_MSM/2ms/tica_transformed_data_23_56_combined_80.pkl','rb'))

cluster_obj = MiniBatchKMeans(n_clusters=350,  # place k cluster centers
                              batch_size=10000,
                              max_iter=50,
                            init_strategy='kmeans++',  # kmeans++ initialization strategy
                            n_jobs=24).fit_fetch(np.concatenate(transformed_data))

dtrajs = [cluster_obj.transform(x) for x in transformed_data]
pickle.dump(dtrajs, open('/home/pdb3/SMO/6XBL_CHL/Analysis/remaking_MSM/2ms/clus_tica_dtrajs/clus_tica_dtrajs_combined_350_80.pkl','wb'))

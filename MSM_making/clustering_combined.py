clusters = list(range(100, 501, 50))+list(range(600, 1001, 100))
cutoffs = [50, 65, 80, 95]

nodes = ['all.q@compute-0-0.local','all.q@compute-0-1.local','all.q@compute-0-10.local','all.q@compute-0-11.local','all.q@compute-0-12.local','all.q@compute-0-13.local','all.q@compute-0-14.local','all.q@compute-0-2.local','all.q@compute-0-3.local','all.q@compute-0-4.local','all.q@compute-0-5.local','all.q@compute-0-6.local','all.q@compute-0-7.local','all.q@compute-0-8.local','all.q@compute-0-9.local','analysis.q@compute-0-15.local','analysis.q@compute-1-0.local','analysis.q@compute-1-1.local']*10

l=0

for k in clusters:
    for ct in cutoffs:
        with open(f'./clustering/clustering_combined_{k}_{ct}.py','w') as f:
            f.write(f'''from pdb3 import lsext
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

transformed_data = pickle.load(open('/home/pdb3/SMO/6XBL_CHL/Analysis/remaking_MSM/2ms/tica_transformed_data_23_56_combined_{ct}.pkl','rb'))

cluster_obj = MiniBatchKMeans(n_clusters={k},  # place k cluster centers
                              batch_size=10000,
                              max_iter=50,
                            init_strategy='kmeans++',  # kmeans++ initialization strategy
                            n_jobs=24).fit_fetch(np.concatenate(transformed_data))

dtrajs = [cluster_obj.transform(x) for x in transformed_data]
pickle.dump(dtrajs, open('/home/pdb3/SMO/6XBL_CHL/Analysis/remaking_MSM/2ms/clus_tica_dtrajs/clus_tica_dtrajs_combined_{k}_{ct}.pkl','wb'))''')
        with open(f'./clustering/clustering_combined_{k}_{ct}.bash','w') as f:
            f.write(f'''#$ -S /bin/bash    # Set shell to run job
#$ -q {nodes[l]}        # Choose queue to run job in
#$ -pe smp 24      # Request processors, other options include onenode, distribute, orte
#$ -cwd            # Run job from my current working directory

python /home/pdb3/SMO/6XBL_CHL/Analysis/remaking_MSM/2ms/clustering/clustering_combined_{k}_{ct}.py''')
        l+=1

from pdb3 import lsext
from tqdm import tqdm
import deeptime
import numpy as np
import matplotlib.pyplot as plt
import tol_colors as tc
cmap=tc.tol_cmap('rainbow_PuBr')
from pdb3 import lsext
import mplhelv
from tqdm import tqdm
from deeptime.decomposition import TICA
from deeptime.clustering import KMeans
import pickle
from deeptime.util import energy2d
from deeptime.plots import plot_energy2d
tol_cmap = tc.tol_cmap('rainbow_PuBr')

total_data = pickle.load(open('./totdist_2ms.pkl','rb'))
cutoffs = [50,65,80,95]
for tno, t in tqdm(enumerate(total_data), total=len(total_data)):
        total_data[tno] = t[:,:89]
for ct in tqdm(cutoffs):
        tica = TICA(lagtime=300,var_cutoff=ct/100).fit_fetch(np.concatenate(total_data))
        projection = [tica.transform(x) for x in total_data]
        pickle.dump(projection, open(f'./tica_transformed_data_23_56_combined_{ct}.pkl','wb'))

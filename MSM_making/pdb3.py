import numpy as np
import numpy.linalg as la
import os
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib as mpl
def lsext(path,ext,extra=None,sort=False,preapp=True,extranot=None,nat=True,pathcheck=False,abs=True,extraand=None):
    extens = [j for j in os.listdir(path) if '{}'.format(ext) in j]
    if extra != None:
        if isinstance(extra,str):
            extens = [j for j in extens if '{}'.format(extra) in j]
        elif isinstance(extra,list):
            extens2 = [j for j in extens if '{}'.format(extra[0]) in j]
            for extraelem in extra[1:]:
                extens2.extend([j for j in extens if '{}'.format(extraelem) in j])
            extens = extens2
    if extraand!=None:
        for extraelem in extraand:
            extens = [j for j in extens if f'{extraelem}' in j]
    if extranot != None:
        if isinstance(extranot,str):
            extens = [j for j in extens if '{}'.format(extranot) not in j]
        elif isinstance(extranot,list):
            extens2 = [j for j in extens if '{}'.format(extranot[0]) not in j]
            #print(len(extens),len(extens2))
            for extraelem in extranot[1:]:
                extens2 = [j for j in extens2 if '{}'.format(extraelem) not in j]
                #print(len(extens),len(extens))
            extens = extens2[:]
    if preapp:
        extens = [path+j for j in extens]
    if sort:
        extens=sorted(extens)
    if nat:
        extens=natsorted(extens)
    if pathcheck and preapp:
        extens = [j for j in extens if os.path.exists(j)]
    if abs and preapp:
        extens = [os.path.abspath(j) for j in extens]
    leng = len(extens)
    return (extens,leng)
def prompt(items,delimiter='_'):
    prompt='{}'.format(delimiter).join(sorted(items)[0].split('{}'.format(delimiter))[:-1])+'{}'.format(delimiter)
    return prompt
def angle(c1,c2,c3):
    vec1 = c2-c1
    vec2 = c2-c3
    cos = 180/np.pi*np.arccos(np.dot(vec1,vec2)/(la.norm(vec1)*la.norm(vec2)))
    return cos
def dihed(c1,c2,c3,c4):
    vec1=c2-c1
    vec2=c3-c2
    vec3=c4-c3
    n1=np.cross(vec2,vec1)
    n1=n1/la.norm(n1)
    n2=np.cross(vec3,vec2)
    n2=n2/la.norm(n2)
    vec2=vec2/la.norm(vec2)
    m1=np.cross(vec2,n1)
    x=np.dot(n1,n2)
    y=np.dot(m1,n2)
    di=180/np.pi*math.atan2(y,x)
    return di
def distance(c1,c2):
    dist=[]
    for j,k in zip(c1,c2):
        dist.append(np.array([la.norm(j-k)]))
    return np.array(dist)
def plane(p1,p2,p3): 
    ''' 
        Returns coefficients [a,b,c,d] for equation of plane of the form
        ax + by + cz + d = 0
        Given three points p1, p2, p3
    '''
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    return np.array([a,b,c,-d])
def plane_traj(traj):  
    ''' 
        Returns coefficients [[a,b,c,d]....[a,b,c,d]] for equation of plane of the form
        ax + by + cz + d = 0
        Given a trajectory array of shape [any, 3, 3]
    '''
    if traj.ndim==3:
        traj_planes=np.empty([len(traj),4])
        for i,j in enumerate(traj):
            traj_planes[i] = plane(j[0],j[1],j[2])
        return traj_planes
    else:
        raise ShapeError('Shape must be 3 dimensional')
def plane_angle(p1,p2,deg=False): #p1, p2, are numpy arrays (len 4 each) for the 2 planes.
    a1,b1,c1,d1 = p1
    a2,b2,c2,d2 = p2
    num = a1*a2+b1*b2+c1*c2
    den = la.norm(p1[:3])*la.norm(p2[:3])
    ang = np.arccos(num/den)
    if deg:
        ang = 180/np.pi*ang
    return ang
def plane_angle_traj(traj1,traj2,deg=False):
    ''' 
        Returns angles for planes defined in traj1 and traj2
        Given a plane array of shape [any, 4]
    '''
    if traj2.ndim==1 and traj1.ndim!=1:
        traj2 = np.array([traj2 for j in range(len(traj1))])
    plane_ang = np.empty([len(traj1),1])
    for i,p1,p2 in zip(range(len(traj1)),traj1,traj2):
        plane_ang[i] = plane_angle(p1,p2)
    if deg:
        plane_ang = 180/np.pi*plane_ang
    return plane_ang

def acute_plane_angle(p1,p2): #p1, p2, are numpy arrays (len 4 each) for the 2 planes.
    a1,b1,c1,d1 = p1
    a2,b2,c2,d2 = p2
    num = a1*a2+b1*b2+c1*c2
    den = la.norm(p1[:3])*la.norm(p2[:3])
    ang = min(np.arccos(np.abs(num/den)), np.pi - np.arccos(np.abs(num/den)))
    return ang
def distance_point_from_plane(p1,p2):
    x1,y1,z1 = p1
    a,b,c,d = p2
    num = np.abs(a*x1+b*y1+c*z1+d)
    den =la.norm(p2[:3])
    return num/den
def free_energy_plot(x_data,y_data,fig,axs,cmap='jet',xmul=1,ymul=1,nbins=200,xlim=None,ylim=None,weights=None,return_c=False):    
    x_data=x_data*xmul
    y_data=y_data*ymul
    R = 0.001987
    T = 310
    fig_wid = 10
    fig_hig = 7
    Max_energy = 5
    
    x_bins=nbins
    y_bins=nbins
    
    x_data_min =  np.min(x_data)
    y_data_min =  np.min(y_data)
    x_data_max =  np.max(x_data)
    y_data_max =  np.max(y_data)
    
    x_hist_lim_low =  x_data_min -0.5
    y_hist_lim_low =  y_data_min -0.5
    x_hist_lim_high = x_data_max +0.5
    y_hist_lim_high = y_data_max +0.5
    
    # if weight:
    #     weights=np.load(weights)
        
    xspace = abs(np.min(x_data) - np.max(x_data))/10
    yspace = abs(np.min(y_data) - np.max(y_data))/10
    
    x_lim_low = np.min(x_data) - xspace
    x_lim_high = np.max(x_data) + xspace 

    y_lim_low = np.min(y_data) - yspace
    y_lim_high = np.max(y_data) + yspace

    hist= np.histogram2d(x_data,y_data, bins=[x_bins,y_bins],
				 range = [[x_hist_lim_low,x_hist_lim_high],[y_hist_lim_low,y_hist_lim_high]],
				 density= True,weights=weights)
    prob_density = hist[0]
    xedge = hist[1]
    yedge = hist[2]
    x_bin_size = xedge[1]-xedge[0]
    y_bin_size = yedge[1]-yedge[0]
    free_energy = -R*T*np.log(prob_density*x_bin_size*y_bin_size)
    min_free_energy = np.min(free_energy)
    delta_free_energy = free_energy - min_free_energy
    xx = [(xedge[i]+xedge[i+1])/2 for i in range(len(xedge)-1)]
    yy = [(yedge[i]+yedge[i+1])/2 for i in range(len(yedge)-1)]
    #fig, axs = plt.subplots(1,1,figsize=(fig_wid,fig_hig))
    contours = np.linspace(0,Max_energy,5)
    # cdl = axs.contour(xx,yy,delta_free_energy.T,levels=6,colors='k',linewidths = 0.2)
    cd = axs.contourf(xx,yy,delta_free_energy.T,np.linspace(0,Max_energy,30), vmin=0.0, vmax=Max_energy,cmap=cmap)
    # cd = axs.hexbin(xx,yy,delta_free_energy.T,np.linspace(0,Max_energy,30), vmin=0.0, vmax=Max_energy,cmap=cmap)
    cbar = fig.colorbar(cd,ticks=range(int(Max_energy)+1),ax=axs)
    cbar.ax.set_ylabel('Free Energy (kcal/mol)',fontsize=12)
    cbar.ax.set_yticklabels(range(int(Max_energy)+1))
    cbar.ax.tick_params(axis='y',labelsize=12)
    # axs.set_xticks(range(int(x_lim_low),int(x_lim_high),5))
    # axs.set_xticklabels(range(int(x_lim_low),int(x_lim_high),5))
    #np.save(f'{i}_vs_{j}_xticks.npy',)
    # axs.set_yticks(np.around(np.floor(np.arange(y_lim_low,y_lim_high,(y_lim_high-y_lim_low)/5)),2))
    # axs.set_yticklabels(np.around(np.floor(np.arange(y_lim_low,y_lim_high,(y_lim_high-y_lim_low)/5)),2))
    #np.save(f'{i}_vs_{j}_yticks.npy',np.around(np.floor(np.arange(y_lim_low,y_lim_high,(y_lim_high-y_lim_low)/5)),2))
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    axs.grid(ls='-',lw=0.2)
    if xlim!=None:
        axs.set_xlim(xlim[0],xlim[1])
    if ylim!=None:
        axs.set_ylim(ylim[0],ylim[1])
    return_list = [fig,axs]
    if return_c:
        return_list.extend([xx,yy,delta_free_energy.T])
    return return_list
def get_RESNO(mdtrajresno,csv='/home/pdb3/SMO/APO/Analysis/SMO_BW_Numbering.csv'):
    import pandas as pd
    df = pd.read_csv(csv)
    f = df[df['mdtraj_num_active']==mdtrajresno].Resno
    FullResno = str(f).split()[1]
    return FullResno
def get_helix(mdtrajresno,csv = '/home/pdb3/SMO/APO/Analysis/SMO_BW_Numbering.csv',BW=False):
    import pandas as pd
    df = pd.read_csv(csv)
    f = df[df['mdtraj_num_active']==mdtrajresno].Location
    helixno = str(f).split()[1]
    if BW:
        g = df[df['mdtraj_num_active']==mdtrajresno].BW
        BW = str(g).split()[1]
        return helixno,BW
    return helixno
def get_RESNAME(mdtrajresno,csv = '/home/pdb3/SMO/APO/Analysis/SMO_BW_Numbering.csv'):
    import pandas as pd
    df = pd.read_csv(csv)
    f = df[df['mdtraj_num_active']==mdtrajresno].Residue
    FullResno = str(f).split()[1]
    return FullResno
SMO_APO_totdist = '/home/pdb3/SMO/APO/Analysis/totdistcombined_extra.npy';
SMO_APO_nc = '/home/pdb3/ds02/SMO/APO/Analysis/nc/';
SMO_APO_dcd = '/home/pdb3/SMO/APO/Analysis/dcd/';
parm_6XBL_APO = '/home/pdb3/SMO/APO/Analysis/6XBL_Apo_HMass.parm7';
parmstrip_6XBL_APO = '/home/pdb3/SMO/APO/Analysis/6XBL_Apo_HMass_stripped.parm7';
parm_5L7D_APO = '/home/pdb3/SMO/APO/Analysis/5L7D_Apo_HMass.parm7';
parmstrip_5L7D_APO = '/home/pdb3/SMO/APO/Analysis/5L7D_Apo_HMass_stripped.parm7';
parmstrip_6XBL_CHL='/home/pdb3/SMO/6XBL_CHL/Analysis/6XBL_CHL3_HMass_stripped.parm7';
parmstrip_6XBL_SAG='/home/pdb3/SMO/6XBL_SAG/Analysis/6XBL_SAG_HMass_stripped_vmd2.psf';
parmstrip_5L7D_SANT1='/home/pdb3/SMO/5L7D_SANT1/Analysis/5L7D_SANT1_HMass_stripped.parm7';
res_subs_dict={ 'GLY':'G',
			'ALA':'A',
			'PRO':'P',
			'VAL':'V',
			'LEU':'L',
			'ILE':'I',
			'MET':'M',
			'PHE':'F',
			'TYR':'Y',
			'TRP':'W',
			'SER':'S',
			'THR':'T',
			'CYS':'C',
			'ASN':'N',
			'GLN':'Q',
			'LYS':'K',
			'ARG':'R',
			'HIS':'H',
			'ASP':'D',
			'GLU':'E'  }

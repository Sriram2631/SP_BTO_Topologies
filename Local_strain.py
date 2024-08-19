#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:54:35 2024

@author: sriram
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import ase
from ase import Atoms
from ase.cell import Cell
from ase.build import make_supercell
from ase.visualize import view
from ase.io import read, write
import pandas as pd
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter
Ref_cubic_file = '/home/sriram/CODES/Ref_structures/Cubic.vasp'
#Ref_cubic_file = 'Amm2.vasp'
New_struct = 'DFT_Final.vasp'
fig_path = 'Etaxy_DFT_Skrm.png'
primx = [1.0,0.0,0.0]
primy = [0.0,1.0,0.0]
primz = [0.0,0.0,1.0]
prim_v = [primx,primy,primz]
sc_size = [7,7,1]


def dist(x,y,vec=False):
    d = np.sqrt( (x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)
    if vec==False:
        return d
    if vec==True:
        vector = [k-j for k,j in zip(y,x)]
        return vector, d

def read_pol(data): 
    x= data.columns.tolist()
    y= data.index.tolist()
    pol_mat = np.zeros((len(x),len(y),1))
    
    for i,j in zip(x,y):
        pol_mat[x.index(i),y.index(j)]= data.loc[i][j]
        
    return pol_mat

def neighbours(pt, Collection, r1=0.1,r2=0.01):
    iselect = []
    for i in range(len(Collection)):
        d = dist(pt,Collection[i])
        if d<=r1 and d>0 and d>r2 :
            iselect.append(int(i))
    return iselect 

center = [30.0,30.0,0.0]

Ref_cell = read(Ref_cubic_file,format='vasp')

repeat = [[sc_size[0],0,0],[0,sc_size[1],0],[0,0,sc_size[2]]]
SC = make_supercell(Ref_cell, repeat)
atoms = SC.get_atomic_numbers()
plot_list = list(filter(lambda x: atoms[x] == 56, range(len(atoms))))
plot_pos = SC.get_positions()[plot_list]
dfref = pd.DataFrame(plot_pos)
dfref.columns = ["x","y","z"]

pos = Ref_cell.get_positions()
ref_angles = Ref_cell.cell.cellpar()
xyref = ref_angles[5]
Ba_Ti = dist(pos[0],pos[1])
acell_ref = Ba_Ti*2/(np.sqrt(3))

Final_struct = read(New_struct,format='vasp')
atoms = Final_struct.get_atomic_numbers()

znucl_list = [56]
for znucl in znucl_list:
    etaxy_list = []
    etax_list = []
    etay_list = []
    print("-------------------------Z = ",znucl,"----------------")
    Ba_list = list(filter(lambda x: atoms[x] == znucl, range(len(atoms))))
    Ba_pos = Final_struct.get_positions()[Ba_list]
    
    dfBa = pd.DataFrame(Ba_pos)
    dfBa.columns = ["x","y","z"]
    
    
    
    r = acell_ref*np.sqrt(3)
    
    for center in Ba_pos:
        i_region_Ba = neighbours(center, Ba_pos,r1=r,r2=0)
        Ba_neigh1 = dfBa.iloc[i_region_Ba].to_numpy()
        
        sep = []
        for ele in Ba_neigh1:
            sep.append(dist(ele,center))
        new_center = Ba_neigh1[sep.index(min(sep))]
       
        
        inew_Ba = neighbours(new_center, Ba_pos, r1=r,r2=0)
        separation = []
        direction = []
        Ba_neigh = dfBa.iloc[inew_Ba].to_numpy()
        for ele in Ba_neigh:
            v, d = dist(new_center,ele,vec = True)
            if d>0.1:
                separation.append(d)
                direction.append(v/acell_ref)
                
        distances = []
        
    
        strain = (separation-acell_ref)/acell_ref
        new_vec = []
        
        for i in range(len(separation)):
            if np.isclose(np.dot(direction[i],[1,1,0]),1.0,atol=0.2):
                if np.isclose(np.dot(direction[i],primx),1.0,atol=0.1):
                    etax = strain[i]
                    vecx = direction[i]
                    
                if np.isclose(np.dot(direction[i],primy),1.0,atol=0.1): 
                    etay = strain[i]
                    vecy = direction[i]
                    
    
        xyangle = np.rad2deg(np.arccos(np.dot(vecy,vecx)/(np.linalg.norm(vecx)*np.linalg.norm(vecy))))
        etaxy = (xyangle-xyref)*100/xyref
        etaxy_list.append(etaxy)
        etax_list.append(etax)
        etay_list.append(etay)
        
    fig,ax = plt.subplots() 
    dfref['etax'] = etax_list
    dfref['etay'] = etay_list
    dfref['etaxy']= etaxy_list
    
    strain_matrix = dfref.pivot_table(index='y',columns='x',values='etaxy')
    df3_smooth = gaussian_filter(strain_matrix, sigma=0.00002)
    
    ax = sns.heatmap(df3_smooth, vmin=np.min(df3_smooth), vmax=np.max(df3_smooth), cmap="RdYlGn", annot=False, fmt="0.0f", cbar_kws={'label': '$\epsilon_{xy}$ (%)'})

    ax.invert_yaxis()
    plt.title('Local Shear strain  - $\epsilon_{xy}$')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fig_path,dpi=300)
    

    
    
    
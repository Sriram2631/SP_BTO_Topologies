#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:20:06 2024

@author: sanand
"""

import ase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from ase.io import *
from ase.visualize import view
from ase.build import make_supercell
from scipy import spatial
import math
import shelve

Ref_cubic_file = 'Ref_structures/Cubic.vasp'
center_Pz=True
Write_file = True
id_center = False
Ti_down = True
rotate_bw = False
prefix = '4_SC_29'
theta = -0.005
sc_size = [29,29,1]
rotation_center_init = [0.5,0.5,0.5] #[0.5,0.5,0.5]
#264706,0.264706
r1 = 0.1
r2 = 0.05
Pz_for_center = [0.000,0.000,0.008]

nshell_up = 0.5
nshell_rot = 2.9
nshell_plane = 1
sym_add = np.array([1,1,1])
symadd = False




fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def dist(x,y):
    d = np.sqrt( (x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2  )
    return d


def neighbours(pt, Collection, r1=0.1,r2=0.01,symadd=False):
    iselect = []
    pt = np.array(pt)
    if symadd:
        ptlist=[pt,pt+sym_add]
    else:
        ptlist=[pt]
    for pt_ind in ptlist:
        print("=========================================================================================")
        print("Neighbour count of", pt_ind, "with radii", r1,r2,"are: (x,y,z,id)")
        for i in range(len(Collection)):
            d = dist(pt_ind,Collection[i])
            if d<=r1 and d>0 and d>=r2 :
                iselect.append(int(i))
        #        print(Collection[i], i)
        print(len(iselect))
    if len(iselect):
        return iselect
        
                       
def rotate(origin, point, angle,up=0):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    Angle in radians.
    """
    ox, oy = origin[:2]
    px, py = point[:2]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    
    return [qx, qy,0.50000+up]

def move(point,up=True,P=0.002):
    if up:
        point[2]=point[2]+P
    else:
        point[2]=point[2]-P
    return point


repeat = [[sc_size[0],0,0],[0,sc_size[1],0],[0,0,sc_size[2]]]

ref = read(Ref_cubic_file,format='vasp')
SC = make_supercell(ref, repeat)
acell = ref.get_cell_lengths_and_angles()[0]


addr1 = nshell_plane*np.sqrt(2)/sc_size[0]
# addr1 = nshell_down*np.sqrt(2)/sc_size[0]
rcenter = nshell_up*np.sqrt(2)/sc_size[0]
r1 = rcenter + nshell_rot*acell*np.sqrt(2)/(sc_size[0]*acell)
r2 = rcenter
atoms = SC.get_atomic_numbers()
res_list = list(filter(lambda x: atoms[x] == 22, range(len(atoms))))
allpos = SC.get_scaled_positions()
Ti_pos = SC.get_scaled_positions()[res_list]

for ele in Ti_pos:
    if not np.isclose(ele[2],0.5):
        print(ele)

Tix,Tiy,Tiz = map(list, zip(*Ti_pos))

df = pd.DataFrame(Ti_pos)
df.columns = ["x","y","z"]

i_altered = []
if id_center:
    i_for_z = neighbours(rotation_center_init, Ti_pos,r1 = rcenter ,r2=0.000)
    Ti_for_z = df.iloc[i_for_z].to_numpy()
    rotation_center = Ti_for_z[0]
    print('Shifting center to ',rotation_center)
else:
    rotation_center = rotation_center_init

inew = neighbours(rotation_center, Ti_pos,r1 = r1,r2=r2)
for ele in inew:
    i_altered.append(ele)
Tixselect,Tiyselect,Tizselect = map(list, zip(*Ti_pos[inew]))

#ax.scatter(Tixselect,Tiyselect,Tizselect,color='black')
Ti_to_move = df.iloc[inew].to_numpy()

Ti_after_move = []

for ele in Ti_to_move:
    new_coord = rotate(rotation_center,ele,angle = theta,up=0.002)
    Ti_after_move.append(new_coord)

with open('/home/sriram/rotate_ti.data','w') as F:
    for ele in inew:
        print(2+5*ele,file=F)

Tixnew,Tiynew,Tiznew = map(list, zip(*Ti_after_move)) 
ax.scatter(Tixnew,Tiynew,Tiznew,color='r')
dfnew = df.copy()
dfnew.iloc[inew]=Ti_after_move

if center_Pz:
    i_for_z = neighbours(rotation_center, Ti_pos,r1 = rcenter ,r2=0.000)   # Uncomment these lines
                                                                           # if location of the rotation center
    Ti_for_z = df.iloc[i_for_z].to_numpy()                                 # is precisely what is given initially
    Ti_z_move = np.array([elecenter + Pz_for_center for elecenter in Ti_for_z])
    Tix_z,Tiy_z,Tiz_z = map(list, zip(*Ti_z_move))
    ax.scatter(Tix_z,Tiy_z,Tiz_z,color='black')
    dfnew.iloc[i_for_z]=Ti_z_move
    for ele in i_for_z:
        i_altered.append(ele)



if rotate_bw:
    i_inter = neighbours(rotation_center, Ti_pos,r1 = r1+addr1,r2=r1)
    Tixselect_inter,Tiyselect_inter,Tizselect_inter = map(list, zip(*Ti_pos[i_inter]))
    Ti_after_inter_move = []
    Ti_inter_move = df.iloc[i_inter].to_numpy()
    for ele in Ti_inter_move:
        new_coord = rotate(rotation_center,ele,angle = theta,up=-0.004)
        Ti_after_inter_move.append(new_coord)
    Tixselect,Tiyselect,Tizselect = map(list, zip(*Ti_after_inter_move))
    ax.scatter(Tixselect,Tiyselect,Tizselect,color='blue')
    dfnew.iloc[i_inter]=Ti_after_inter_move
    for ele in i_inter:
        i_altered.append(ele)


if symadd:
    theta = -1*theta
    rotation_center_init2 = sym_add - rotation_center
    rotation_center = rotation_center_init2
        
    
    inew1 = neighbours(rotation_center, Ti_pos,r1 = r1,r2=r2)
    for ele in inew1:
        i_altered.append(ele)
    Tixselect,Tiyselect,Tizselect = map(list, zip(*Ti_pos[inew1]))
    Ti_to_move1 = df.iloc[inew1].to_numpy()

    Ti_after_move1 = []

    for ele in Ti_to_move1:
        new_coord = rotate(rotation_center,ele,angle = theta,up=0.000)
        Ti_after_move1.append(new_coord)



    Tixnew,Tiynew,Tiznew = map(list, zip(*Ti_after_move1)) 
    ax.scatter(Tixnew,Tiynew,Tiznew,color='r')
    dfnew.iloc[inew1]=Ti_after_move1

    if center_Pz:
        i_for_z1 = neighbours(rotation_center, Ti_pos,r1 = rcenter ,r2=0.000)
        
        Ti_for_z1 = df.iloc[i_for_z1].to_numpy()
        
        Ti_z_move1 = np.array([elecenter + Pz_for_center for elecenter in Ti_for_z1])
        Tix_z,Tiy_z,Tiz_z = map(list, zip(*Ti_z_move1))
        ax.scatter(Tix_z,Tiy_z,Tiz_z,color='black')
        dfnew.iloc[i_for_z1]=Ti_z_move1
        for ele in i_for_z1:
            i_altered.append(ele)

i_down_new = []
for i in range(len(Ti_pos)):
    if i not in i_altered:
        i_down_new.append(i)
        
if Ti_down:
    i_for_down = i_down_new 
    Ti_for_down= df.iloc[i_for_down].to_numpy()
    Ti_z_down = np.array([elecenter - Pz_for_center for elecenter in Ti_for_down])
    Tixselect,Tiyselect,Tizselect = map(list, zip(*Ti_z_down))
    ax.scatter(Tixselect,Tiyselect,Tizselect,color='black')
    dfnew.iloc[i_for_down]=Ti_z_down

Ti_new_pos = dfnew.to_numpy()
allpos_new = allpos.copy()
allpos_new[res_list] = Ti_new_pos
### Selection 1 ##########

x = np.linspace(0, 1, 500)
y = np.linspace(0, 1, 500)

x, y = np.meshgrid(x, y)
z= 0*x+0*y+0.5 


ax.plot_wireframe(x, y, z,alpha = 0.2)

SC.set_scaled_positions(allpos_new)

Write_info = {'Ref_file':Ref_cubic_file,
               'Center Pz': center_Pz,
               'theta': theta,
               'SC_size':sc_size,
               'rotation_center':[rotation_center_init,rotation_center],
               'n shell up': nshell_up,
               'n shells rot':nshell_rot,
               'n shell plane':nshell_plane,
               # 'n shell down':nshell_down,
               'r1':r1,
               'r2':r2}
if Write_file:
    if symadd:
        prefix = '/Pairs/' + prefix
    Outfile='/home/sriram/Distorting_structures/Code_generated/Skrm2/'+prefix+'.vasp'
    Figsave = '/home/sriram/Distorting_structures/Code_generated/Skrm2/'+prefix+'.png'
    Shelve_file = '/home/sriram/Distorting_structures/Code_generated/Skrm2/'+prefix+'.out'
    
    write(Outfile,SC,format='vasp') # <--- To write
    plt.savefig(Figsave,dpi=300)
    with open(Shelve_file, 'w') as f:  
        for key, value in Write_info.items():  
            f.write('%s:%s\n' % (key, value))
    
    #Shelve_all()
#SC.set_scaled_positions(Ti_new_pos)




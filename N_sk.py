#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:14:37 2024

@author: sanand
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def dist(x,y):
    d = np.sqrt( (x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2  )
    return d

def Normalise(pol_mat,dim):
    pol_mat_normed = np.zeros((dim[0],dim[1],dim[2],3))
    for k in range(dim[2]):
        for j in range(dim[1]):
            for i in range(dim[0]):
                pol_mat_normed[i,j,k] = pol_mat[i,j,k]/np.linalg.norm(pol_mat[i,j,k])
    return pol_mat_normed

def plot_P(pol_mat,dim,scale=10,path=None):
    pol_mat_og = pol_mat.copy()
    fig,ax = plt.subplots() 
    plt.xticks([])
    plt.yticks([])
    for zd in range(dim[2]):
        qq=ax.quiver(np.transpose(pol_mat_og[:,:,zd,0]), np.transpose(pol_mat_og[:,:,zd,1]),scale=scale)
        cax2 = ax.imshow(np.transpose(pol_mat_og[:,:,zd,2]), origin='lower', cmap=plt.cm.rainbow,vmin=-0.3,vmax=0.3,
                                alpha=1, interpolation=Fig_specs['intp'])
        cbar2 = plt.colorbar(cax2)
        cbar2.set_label('Pz ($C/m^2$)',fontsize=12,rotation=270)
        plt.savefig(path,dpi=300)
        plt.show()
        plt.close()
    

def plot_Q(q,dim,x,y,savefig=False,varname='q',path=None):
    for zd in range(dim[2]):
        fig,ax = plt.subplots() 
        plt.xticks([])
        plt.yticks([])
        val = 0
        plt.scatter(x,y,[11 for ele in x],color='black',alpha=0.2)
        for h in range(dim[0]):
            for j in range(dim[1]):
                #if h in [6,7,8,9,10,11,12,13,14] and j in [6,7,8,9,10,11,12,13,14]:
                #if (h,j) in [(1,4),(4,1),(4,4),(4,7),(4,8),(8,4),(7,4)]: # or -h+4j<32:
                val+=q[h,j,zd]     
               
        
        string = ' '
        
        cax = ax.imshow(np.transpose(q[:,:,zd]), origin='lower', cmap=Fig_specs['cmap'], #plt.cm.rainbow,
                                  alpha=1.00,interpolation=Fig_specs['intp'])
        cbar = plt.colorbar(cax)
        cbar.set_label(varname, fontsize=12,rotation=0)
        if dim[2]>1:
            text_to_write = "z plane: "+str(zd) + "\t $\Sigma \Sigma$"+varname+" = "+ str(np.round(val,3))
            ax.text(max(x)/6, max(y)+0.7,text_to_write, style='oblique')
        else:
            text_to_write = "\t $\Sigma \Sigma$"+varname+" = "+ str(np.round(val,3))
            ax.text(max(x)/6, max(y)+0.7,text_to_write, style='oblique')
        
        ax.text(max(x)/3.5, min(y)-1,string, style='normal')
        
        if savefig:
            file_to_save = path
            plt.savefig(path,dpi = 300)
        else:
            plt.show()
        plt.close()
        print("Zd plane: " ,zd, "Sum of all "+varname+": ", round(val,4))
        
def q_finite_diff(pol_mat,dim):
    q = np.zeros((dim[0],dim[1],dim[2]))
    pol_mat_norm = Normalise(pol_mat,dim) #[i,j,k]/np.linalg.norm(pol_mat[i,j,k])
    for k in range(dim[2]):
        for j in range(dim[1]):
            for i in range(dim[0]):
                dpdx_pt = [0,0,0]
                dpdy_pt = [0,0,0]
                if i==0:                                            #PBC
                    dpdx_pt = (pol_mat[i+1,j,k]-pol_mat[i-1,j,k])/2
                if i==dim[0]-1:                                     #PBC 
                    dpdx_pt = (pol_mat[0,j,k]-pol_mat[i-1,j,k])/2
                if j==0:                                            #PBC
                    dpdy_pt = (pol_mat[i,j+1,k]-pol_mat[i,j-1,k])/2
                if j==dim[1]-1:                                     #PBC
                    dpdy_pt = (pol_mat[i,0,k]-pol_mat[i,j-1,k])/2
                if i in range(1,dim[0]-1) and j in range(1,dim[1]-1):
                    dpdx_pt = (pol_mat_norm[i+1,j,k]-pol_mat_norm[i-1,j,k])/2
                    dpdy_pt = (pol_mat_norm[i,j+1,k]-pol_mat_norm[i,j-1,k])/2
                cross_pt = np.cross(dpdx_pt,dpdy_pt)
                q[i,j,k] = np.dot(pol_mat_norm[i,j,k],cross_pt)/(4*np.pi)
    return q

def q_solid_angle(pol_mat,dim):
    q = np.zeros((dim[0],dim[1],dim[2]))
    pol_mat_norm = Normalise(pol_mat,dim)
    for k in range(dim[2]):
        for i in range(dim[0]):
            for j in range(dim[1]):
                a1 = pol_mat_norm[i,j,k]
                a2 = a1.copy()
                if i==0:                                            #PBC
                    c2 = pol_mat_norm[dim[0]-1,j,k]
                else:
                    c2 = pol_mat_norm[i-1,j,k]
                if i==dim[0]-1:                                     #PBC 
                    c1 = pol_mat_norm[0,j,k]
                else:
                    c1 = pol_mat_norm[i+1,j,k]
                if j==0:                                            #PBC
                    b1 = pol_mat_norm[i,dim[1]-1,k]
                else:
                    b1 = pol_mat_norm[i,j-1,k]
                if j==dim[1]-1:                                     #PBC
                    b2 = pol_mat_norm[i,0,k]
                else:
                    b2 = pol_mat_norm[i,j+1,k]
                
                if i in range(1,dim[0]-1) and j in range(1,dim[1]-1):
                    b1 = pol_mat_norm[i,j-1,k]
                    c1 = pol_mat_norm[i+1,j,k]
                    b2 = pol_mat_norm[i,j+1,k]
                    c2 = pol_mat_norm[i-1,j,k]
                    
                sign11 = np.cross(b1,c1)
                sign1 = np.dot(a1,sign11)
                sign21 = np.cross(b2,c2)
                sign2 = np.dot(a2,sign21)
                
                nr11 = np.dot(a1,b1)      
                nr12 = np.dot(b1,c1)     
                nr13 = np.dot(c1,a1)      
                nr1 = 1+nr11+nr12+nr13
                dr1 = np.sqrt(2*(1+nr11)*(1+nr12)*(1+nr13))
                Al1 = 2*np.arccos(nr1/dr1)
                Al1_w_sign = math.copysign(Al1,sign1)
            
                nr21 = np.dot(a2,b2)      
                nr22 = np.dot(b2,c2)      
                nr23 = np.dot(c2,a2)     
                nr2 = 1+nr21+nr22+nr23
                dr2 = np.sqrt(2*(1+nr21)*(1+nr22)*(1+nr23))
                Al2 = 2*np.arccos(nr2/dr2)
                Al2_w_sign = math.copysign(Al2,sign2)
                                    
                q[i,j,k] = (Al1_w_sign+Al2_w_sign)/(4*np.pi)
    return q

def read_pol(filename, delimiter):        
    data = pd.read_csv(filename,delimiter=delimiter, header=None) 
    data.columns = ['x','y','z','Px','Py','Pz']
    
    x = data['x'].copy().to_numpy()
    y = data['y'].copy().to_numpy()
    z = data['z'].copy().to_numpy()
    dim = [len(np.unique(x)),len(np.unique(y)),len(np.unique(z))]
    pol_mat = np.zeros((dim[0],dim[1],dim[2],3))
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                pol_mat[i,j,k] = data.loc[(data['x']==i) & (data['y']==j) & (data['z']==k),['Px','Py','Pz']].copy().to_numpy()
    return pol_mat, dim, x,y,z       

 

def Helicity(pol_mat,dim):
    H = np.zeros((dim[0],dim[1],dim[2]))
    # variable H
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                if i==0:                                            #PBC
                    d = pol_mat[dim[0]-1,j,k]
                else:
                    d = pol_mat[i-1,j,k]
                if i==dim[0]-1:                                     #PBC 
                    c = pol_mat[0,j,k]
                else:
                    c = pol_mat[i+1,j,k]
                if j==0:                                            #PBC
                    b = pol_mat[i,dim[1]-1,k]
                else:
                    b = pol_mat[i,j-1,k]
                if j==dim[1]-1:                                     #PBC
                    a = pol_mat[i,0,k]
                else:
                    a = pol_mat[i,j+1,k]
                
                
                    
                curl_i = (a[2] - b[2])/2 - 0
                curl_j = 0 - (c[2]-d[2])/2
                curl_k = (c[1]-d[1])/2 - (a[0]-b[0])/2 #d/dz = 0 since peridic along z, need to change for lattices with more layers!!!

                curl_pt = np.array([curl_i,curl_j,curl_k])
                H_pt = np.dot(pol_mat[i,j,k],curl_pt)
                H[i,j,k] = H_pt
    val = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                val+=H[i,j,k]
                if i>0 and j>0 and i<dim[0]-1 and j<dim[1]-1:
                    if np.isclose(H[i,j,k],0.0):
                        print(i,j,k)
    return H, val

def Curl(pol_mat,dim):
    H = np.zeros((dim[0],dim[1],dim[2]))
    # variable H
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                if i==0:                                            #PBC
                    d = pol_mat[dim[0]-1,j,k]
                else:
                    d = pol_mat[i-1,j,k]
                if i==dim[0]-1:                                     #PBC 
                    c = pol_mat[0,j,k]
                else:
                    c = pol_mat[i+1,j,k]
                if j==0:                                            #PBC
                    b = pol_mat[i,dim[1]-1,k]
                else:
                    b = pol_mat[i,j-1,k]
                if j==dim[1]-1:                                     #PBC
                    a = pol_mat[i,0,k]
                else:
                    a = pol_mat[i,j+1,k]
                    

                curl_k = (c[1]-d[1])/2 - (a[0]-b[0])/2

                curl_pt = curl_k
                H_pt = curl_pt
                H[i,j,k] = H_pt
    val = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                val+=H[i,j,k]
                if i>0 and j>0 and i<dim[0]-1 and j<dim[1]-1:
                    if np.isclose(H[i,j,k],0.0):
                        print(i,j,k)
    
    print("Sum of all H: ",val)
    return H, val
                


def measure_radius(pol_mat,dim,center=None):
    r = 0
    if not center:
        center=[int(ele) for ele in [dim[0]/2,dim[1]/2,0]]
    print(center)
    center_P = pol_mat[center[0],center[1],center[2]]
    opp_list = []
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                dotproduct = np.dot(center_P,pol_mat[i,j,k])
                if dotproduct<0:
                    opp_list.append([i,j,k])
    print(opp_list)
    dlist = []
    for ele in opp_list:
        d = dist(center,ele)
        dlist.append(d)
    minval = min(dlist)
    
    #print(opp_list[dlist.index(minval)])
    print('Radius: ',minval,'x the lattice constant')
    print('point: ', opp_list[dlist.index(minval)])
            

Fig_specs = {'dpi': 300,
             'intp': 'lanczos',
             'cmap': 'Spectral_r',
             'P_scale':10}

if __name__ == '__main__':
    start_time = time.time()     
    file = '/home/sriram/SCRATCH/Mount_clusters/Phythema/CODES/Plot_codes/DFT_Final.txt'

    pol_mat,dim,x,y,z = read_pol(file,' ')
    
    path = '10K_new_rev' 
    plot_P(pol_mat,dim,Fig_specs['P_scale'],path=path+'_P.png')
    
    q = q_solid_angle(pol_mat,dim)

    plot_Q(q,dim,x,y, savefig=False,path=path+'_Q.png') 
    
    H, H_val = Helicity(pol_mat,dim)
    plot_Q(H,dim,x,y,varname='H',savefig=False,path=path+'_H.png')
    
    H, H_val = Curl(pol_mat,dim)
    plot_Q(H,dim,x,y,varname='V',savefig=False,path=path+'_V.png')
    
    end_time = time.time()
    
    measure_radius(pol_mat,dim)
    print('Runtime: ',end_time-start_time,'s')

    






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:16:42 2018

@author: Apoorba
#Purpose : Create T**2 control chart
"""
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("%reset -f")

import pandas as pd
import os 
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from scipy.stats import chi2

# Corrlation Switch
Is_Cov =True
os.chdir("/Users/Apoorb/Documents/GitHub/Phase-1-Analysis--Anomaly-Detection")
if Is_Cov:
    dat=pd.read_csv("PrinCompDat_Covar.csv",index_col=0)
    dat=dat.drop("PC4",axis=1)
else:
    dat=pd.read_csv("PrinCompDat_Correl.csv")#Correlation


def Phase1T2(df,iter1,OC_pnt,p=3,alpha=0.05,ulim=50):
    Ind=df.obs
    X=df.drop("obs",axis=1).copy()
    S=  np.cov(X.T) #Alternate way to get covariance
    X_bar= np.array(X.mean(axis=0))
    X=X.values # Get np array
    S_inv=np.linalg.inv(S)
    T2=np.empty((X.shape[0],1))
    for i in range(X.shape[0]):
        tp=(X[i,:]-X_bar).reshape(1,p).dot(S_inv).dot((X[i,:]-X_bar).reshape(p,1))
        tp=tp.reshape(1)
        T2[i]=tp
    #Chisquare (3)
    #alpha=0.05
    dof=p
    UCL=chi2.ppf(1-alpha,dof)
    fig,ax = plt.subplots()
    ax.plot(Ind,T2,marker="o",label="Current Iteration")
    ax.axhline(y=UCL, color='r', linestyle='-')
    ax.plot(OC_pnt.obs,OC_pnt.T2,'*r',label="Out-Control Points from Previous Iteration")
    handles, labels = ax.get_legend_handles_labels()
    # reverse the order
    ax.legend(handles[::-1], labels[::-1],loc='upper right')
    ax.set(xlabel="Observation",ylabel=r"$T^2$",title=r"$T^2$ Chart with $\alpha$=0.05 (Iteration %s)"%iter1)
    ax.set_ylim(0,ulim)
#red_dot, = plt.plot(z, "ro", markersize=15)
## Put a white cross over some of the data.
#white_cross, = plt.plot(z[:5], "w+", markeredgewidth=3, markersize=15)
#
#plt.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])
#    
    plt.close()
    # Get In Control Points
    df1=df.copy()
    df1.loc[:,"UCL"]=UCL
    df1.loc[:,"T2"]=T2.reshape(df1.shape[0])
    df1.loc[:,"delRow"]=np.where(df1['T2']>df1['UCL'],True, False)
    OCpnt_Prev=df1.loc[df1.delRow,["obs","T2"]].copy()
    #Delete the following obs
    df1.loc[df1.delRow,'obs']
    #Deleting the obs
    df1=df1.query('not delRow')
    res=dict()
    res['Fig']=fig
    res['Ndf']=df1
    res['PrevOc']=OCpnt_Prev
    return res


#Res1=Phase1T2(df=dat)
#Res1['Fig']
#dat2=Res1['Ndf'].loc[:,["PC1","PC2","PC3"]]
#Res2=Phase1T2(df=dat2)
#Res2['Fig']
#dat3=Res2['Ndf'].loc[:,["PC1","PC2","PC3"]]
#Res3=Phase1T2(df=dat3)
#Res3['Fig']
#dat4=Res3['Ndf'].loc[:,["PC1","PC2","PC3"]]
#Res4=Phase1T2(df=dat4)


dat.loc[:,"obs"]=dat.index+1
dat_cpy=dat.copy()
Di_Res=dict()
Di_ResF=dict()
OC_cont=pd.DataFrame({"obs":[],"T2":[]})
i=1
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (15,10),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
while True:
    Res1=Phase1T2(df=dat,iter1=i,OC_pnt=OC_cont)
    Di_Res[i]=Res1["Ndf"]
    Di_ResF[i]=Res1["Fig"]
    OC_cont=pd.concat([OC_cont,Res1["PrevOc"]])
    if(dat.shape[0]==Res1['Ndf'].shape[0]):
        break
    dat=Res1['Ndf'].loc[:,["obs","PC1","PC2","PC3"]].copy()
    i+=1
    
Di_ResF[1]
Di_ResF[2]
Di_ResF[3]
Di_ResF[4]
Di_ResF[5]
Di_ResF[6]
Di_ResF[7]
Di_ResF[8]
Di_ResF[9]

for key in Di_ResF.keys():
    Di_ResF[key].savefig("T2Chart_Iter%s.png"%key)

grid = plt.GridSpec(3, 3, wspace=0.4, hspace=0.3)
Di_ResF[1]
Di_ResF[2]
Di_ResF[3]
Di_ResF[4]
Di_ResF[5]
Di_ResF[6]
Di_ResF[7]
Di_ResF[8]
Di_ResF[9]

#******************************************************************************
IC_dat=np.array(Di_Res[9].obs)
all_dat=np.array(dat_cpy.obs.tolist())
OC_obs=np.setdiff1d(all_dat,IC_dat)


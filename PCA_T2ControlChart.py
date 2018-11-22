#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:13:04 2018
Author: Apoorba
Project: Conduct Phase 1 Analysis
"""

import pandas as pd
import os 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#import plotly
#import plotly.plotly as py
#from plotly.graph_objs import *
#import plotly.tools as tls
#import sys
#sys.path.append("/Users/Apoorb/Documents")
#import Plotly_cred
#import seaborn as sns

#plotly.tools.set_credentials_file(username=Plotly_cred.Plotly_cred1[0],api_key=Plotly_cred.Plotly_cred1[1])

os.chdir("/Users/Apoorb/Documents/GitHub/Phase-1-Analysis--Anomaly-Detection")
X=pd.read_excel("Project_dataset.xlsx",header=None)

# No need to standardize data in this case. But can be done using:
X_std = StandardScaler().fit_transform(X)

#Verbose Method
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

#Direct numpy method
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
#cov_mat = np.cov(X_std.T) #Alternate way to get covariance

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
eig_vecs.shape #209 X 209 as expected


for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0,np.linalg.norm(ev))
print("Everything ok")

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]


# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)



#******************************************************************************
#Pareto Plot
Ncomp=15
Sns_dt=pd.DataFrame({'x1':['PC %s' %(i+1) for i in range(0,Ncomp)],'y1':var_exp[:Ncomp]})
Sns_dt2=pd.DataFrame({'x1':['PC %s' %(i+1) for i in range(0,Ncomp)],'y1':cum_var_exp[:Ncomp]
})
fig,ax = plt.subplots()
ax.bar(Sns_dt['x1'],Sns_dt['y1'],color='blue')
ax.plot(Sns_dt2['x1'],Sns_dt2['y1'],marker='o',color='r',label="cumulative explained variance")
ax.legend(loc='upper left', frameon=False)
ax.set(xlabel="Principle Component",ylabel="Explained variance in %",title="Explained variance by difference principle components")

fig.savefig("ParetoPlt.png")

# Scree Plot
Sns_dt3=pd.DataFrame({'x1':['PC %s' %(i+1) for i in range(0,Ncomp)],'y1':sorted(eig_vals, reverse=True)[:Ncomp]})
fig2,ax = plt.subplots()
ax.plot(Sns_dt3['x1'],Sns_dt3['y1'],color='black')
ax.plot(Sns_dt3['x1'],Sns_dt3['y1'],marker='o',color='black')
ax.set(xlabel="Principle Component",ylabel="Eigen Value (Variance for\n a Principle Component)",title="Scree Plot")
fig2.savefig("ScreePlt.png")

NcompF=5
# Another way to plot
#******************************************************************************
#sns.catplot(x="x1",y="y1",hue=None,kind='bar',data=Sns_dt,color="blue",ax=ax)
#sns.pointplot(x='x1',y='y1',data=Sns_dt2,color='r',ax=ax)
#ax.set(xlabel="Principle Component",ylabel="Explained variance in %")
#plt.close(2)
#plt.show()
# Third way to plot
#******************************************************************************
#trace1 = Bar(
#        x=['PC %s' %i for i in range(1,5)],
#        y=var_exp,
#        showlegend=False)
#
#trace2 = Scatter(
#        x=['PC %s' %i for i in range(1,5)], 
#        y=cum_var_exp,
#        name='cumulative explained variance')
#
#data = Data([trace1, trace2])
#
#layout=Layout(
#        yaxis=YAxis(title='Explained variance in percent'),
#        title='Explained variance by different principal components')
#
#fig1 = Figure(data=data, layout=layout)
#py.iplot(fig1)

# Get the Y
#******************************************************************************
# If only 1st 2 PC are sufficient
#matrix_w = np.hstack((eig_pairs[0][1].reshape(209,1), 
#                      eig_pairs[1][1].reshape(209,1)))
#New Feature Space
#Y=X_std.dot(matrix_w)

#******************************************************************************
pca = PCA(n_components=NcompF)
principalComponents = pca.fit_transform(X_std)
principalDf = pd.DataFrame(data = principalComponents)
principalDf.columns=['PC%s'%(i+1) for i in range(NcompF)]
principalDf.rename({0:"PC1"},axis=1)
principalDf.to_csv("PrinCompDat.csv")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import random
random.seed(67)

import numpy as np
np.random.seed(67)

import matplotlib
matplotlib.use('Agg')

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sompy import SOMFactory
from sklearn.cluster import DBSCAN
from sompy.visualization.mapview import View2D
from sompy.visualization.umatrix import UMatrixView
from sompy.visualization.histogram import Hist2d

import os
import sys

sns.set_style('white')
sns.set_context('notebook', font_scale=2)


df_train = pd.read_csv(os.getenv('PREPARED_TRAINING'))
df_valid = pd.read_csv(os.getenv('PREPARED_VALIDATING'))
df_test = pd.read_csv(os.getenv('PREPARED_TESTING'))

feature_cols = list(df_train.columns[:-1])
target_col = df_train.columns[-1]


X_train = df_train[feature_cols].values
y_train = df_train[target_col].values

X_valid = df_valid[feature_cols].values
y_valid = df_valid[target_col].values

# X_test = df_test[feature_cols].values


sm = SOMFactory.build(X_train, mapsize=[30, 30])
sm.train()


bmu = sm.find_bmu(X_valid)
print(bmu[1].shape)


print(sm.component_names)


xy = sm.bmu_ind_to_xy(bmu[0])
print(xy)


projection = sm.project_data(X_valid)
print(projection)


#sm.predict_by(X_train[:,:-1], y_train[:,-1:])


v = UMatrixView(8, 8, 'SOM', cmap='viridis')
v.show(sm)

plt.savefig(os.path.join(os.getenv('STORING'), 'figure8.png'))


v = View2D(8, 8, 'SOM', cmap='viridis')
v.prepare()
v.show(sm, col_sz=5, cmap='viridis')

plt.rcParams['axes.labelsize'] = 10
plt.savefig(os.path.join(os.getenv('STORING'), 'figure9.png'))


tsne_data = np.load(os.path.join(os.getenv('STORING'), 'tsne_2d_30p.npz'))
tsne_train = tsne_data['train']
tsne_valid = tsne_data['valid']
tsne_test = tsne_data['test']
tsne_all = np.concatenate([tsne_train, tsne_valid, tsne_test], axis=0)


dbscan = DBSCAN(eps=0.1, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None)
dbscan_all = dbscan.fit_predict(tsne_all)


fig = plt.figure(figsize=(24, 24))
ax = fig.add_subplot(111)
ax.scatter(tsne_all[:,0], tsne_all[:,1], c=dbscan_all, cmap='Set3', s=8, alpha=0.8, marker='.', lw=0)

fig.savefig(os.path.join(os.getenv('STORING'), 'figure10.png'))

if bool(int(os.getenv('TSNE_2D_ONLY', '0'))):
    sys.exit()

tsne_data = np.load(os.path.join(os.getenv('STORING'), 'tsne_3d_30p.npz'))
tsne_train = tsne_data['train']
tsne_valid = tsne_data['valid']
tsne_test = tsne_data['test']
tsne_all = np.concatenate([tsne_train, tsne_valid, tsne_test], axis=0)
dbscan = DBSCAN(eps=0.1, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None)
dbscan_all = dbscan.fit_predict(tsne_all)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_all[:,0], tsne_all[:,1], tsne_all[:,2], c=dbscan_all, cmap='Set1', s=10, alpha=1.0, marker='.', lw=0, depthshade=True)

fig.savefig(os.path.join(os.getenv('STORING'), 'figure11.png'))


import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
idx = np.random.choice(len(tsne_all), 10000)
trace = go.Scatter3d(
    x=tsne_all[idx,0],
    y=tsne_all[idx,1],
    z=tsne_all[idx,2],
    mode='markers',
    marker=dict(
        size=4,
        color=dbscan_all[idx],
        colorscale='viridis',
        opacity=0.8
    )
)

plotly.offline.plot(go.Figure(data=[trace], layout=go.Layout(title='tsne-3d-scatter')), filename=os.path.join(os.getenv('STORING'), 'tsne-3d-scatter.html'))

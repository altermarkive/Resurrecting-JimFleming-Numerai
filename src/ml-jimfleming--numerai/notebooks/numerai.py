#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import time

import random
random.seed(67)

import numpy as np
np.random.seed(67)

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os

sns.set_style('white')
sns.set_context('notebook', font_scale=2)

def heatmap(data, fig, ax):
    features = [f for f in list(data) if 'feature' in f]
    features = data[features]
    correlation_matrix = features.corr()
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    cmap = cm.get_cmap('RdBu_r', 30)
    cax = ax.imshow(correlation_matrix, interpolation='nearest', cmap=cmap)
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.title('Feature Correlation')


# Import Data

df_train = pd.read_csv(os.getenv('PREPARED_TRAINING'))
df_valid = pd.read_csv(os.getenv('PREPARED_VALIDATING'))
df_test = pd.read_csv(os.getenv('PREPARED_TESTING'))

feature_cols = list(df_train.columns[:-1])
target_col = df_train.columns[-1]
df_valid[feature_cols] = df_valid[feature_cols].astype(float)

df_train[target_col] = df_train[target_col].astype(np.int32)
df_valid[target_col] = df_valid[target_col].astype(np.int32)

# Visualization

df_plot = pd.melt(df_valid, 'target', var_name='feature')

fig, ax = plt.subplots(figsize=(24, 24))
heatmap(df_valid, fig, ax)

fig.savefig(os.path.join(os.getenv('STORING'), 'figure1.png'))

fig, ax = plt.subplots(figsize=(24, 12))

sns.violinplot(data=df_plot, x='feature', y='value', split=True, hue='target', scale='area', palette='Set3', cut=0, lw=1, inner='quart')
sns.despine(left=True, bottom=True)

ax.set_xticklabels(feature_cols, rotation=90);

fig.savefig(os.path.join(os.getenv('STORING'), 'figure2.png'))

from sklearn.preprocessing import MinMaxScaler, StandardScaler
df_valid_std = df_valid.copy()
df_valid_std[feature_cols] = StandardScaler().fit_transform(df_valid_std[feature_cols])
df_plot_std = pd.melt(df_valid_std, 'target', var_name='feature')
fig, ax = plt.subplots(figsize=(24, 12))
sns.violinplot(data=df_plot_std, x='feature', y='value', split=True, hue='target', scale='area', palette='Set3', cut=0, lw=1, inner='quart')
sns.despine(left=True, bottom=True)
ax.set_xticklabels(feature_cols, rotation=90)

fig.savefig(os.path.join(os.getenv('STORING'), 'figure3.png'))

# df_valid_sample = df_valid.sample(n=100)
# df_valid_sample_x = pd.DataFrame(df_valid_sample, columns=feature_cols)
# df_valid_sample_y = pd.DataFrame(df_valid_sample, columns=[target_col])
# df_valid_sample_y = [(0, 1, 0) if y == 1 else (0, 0, 1) for y in df_valid_sample_y]
#
# plt.rcParams['axes.labelsize'] = 10
# pd.plotting.scatter_matrix(df_valid_sample_x, figsize=(23, 23), marker='o', hist_kwds={'bins': 10}, s=2, alpha=.8)

p = sns.pairplot(df_valid.sample(n=1000), hue='target', vars=feature_cols, size=2)
sns.despine(left=True, bottom=True)

plt.savefig(os.path.join(os.getenv('STORING'), 'figure4.png'))

from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_union, make_pipeline

pipeline = make_pipeline(
    PolynomialFeatures(degree=2),
    PCA(n_components=2),
)

pipeline.fit(df_train[feature_cols].values, df_train[target_col].values)

X_pipeline0 = pipeline.transform(df_valid[df_valid[target_col] == 0][feature_cols].values)
X_pipeline1 = pipeline.transform(df_valid[df_valid[target_col] == 1][feature_cols].values)

import matplotlib.cm as cm

fig, ax = plt.subplots(figsize=(24, 24))
ax.scatter(X_pipeline0[:,0], X_pipeline0[:,1], c=cm.Set3(np.zeros_like(X_pipeline0[:,0])), s=24, lw=0, alpha=0.8, marker='o', label='0')
ax.scatter(X_pipeline1[:,0], X_pipeline1[:,1], c=cm.Set3(np.ones_like(X_pipeline0[:,0])), s=24, lw=0, alpha=0.8, marker='o', label='1')
ax.legend()

fig.savefig(os.path.join(os.getenv('STORING'), 'figure5.png'))

tsne_data = np.load(os.path.join(os.getenv('STORING'), 'tsne_2d_5p.npz'))

fig, ax = plt.subplots(figsize=(25, 25))
plt.scatter(tsne_data['train'][:,0], tsne_data['train'][:,1], c=df_train['target'], cmap='Set3', alpha=0.8, s=4, lw=0)

fig.savefig(os.path.join(os.getenv('STORING'), 'figure6.png'))

from sklearn.manifold import Isomap
from sklearn.pipeline import make_union, make_pipeline

isomap = Isomap()
isomap.fit(df_train[feature_cols].values[:10000])
isomap_train = isomap.transform(df_train[feature_cols].values)
isomap_valid = isomap.transform(df_valid[feature_cols].values)

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=30)
dbscan_train = dbscan.fit_predict(isomap_train)
np.unique(dbscan_train)

fig, ax = plt.subplots(figsize=(24, 24))
plt.scatter(isomap_train[:,0], isomap_train[:,1], c=dbscan_train, cmap='Set1', alpha=0.8, s=8, marker='.', lw=0)

fig.savefig(os.path.join(os.getenv('STORING'), 'figure7.png'))

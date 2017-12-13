# Preface

This repository fork accompanies that YouTube video: https://youtu.be/wju1qtTNiFs

# Numerai

Numerai is a new kind of hedge fund that pays out a reward to data scientists that can improve their meta model.
The idea is to take a large number of uncorrelated high performing models and ensemble them.
Money is divvied out based on leaderboard ranking awarding the top 100.

I tried out an xgboost model a wide and deep model using tensorflow.  Google has used this wide and deep model to
recommend apps on the Google Play store. In this algorithm a wide linear model and a deep feed-forward neural
network are jointly trained.  This approach combines the strengths of memorization and generalization.

Wide and Deep Learning research paper:

https://arxiv.org/abs/1606.07792

The xgboost model was rather poor but the tensorflow model ended up scoring the same logloss as the top 
performing models. 

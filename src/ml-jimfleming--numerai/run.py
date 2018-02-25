#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import sys
import numpy as np


def prepare():
    if 'STORING' not in os.environ:
        os.environ['STORING'] = '/data'
    storing = os.getenv('STORING')
    if 'TRAINING' not in os.environ:
        os.environ['TRAINING'] = os.path.join(
            storing, 'numerai_training_data.csv')
    if 'TESTING' not in os.environ:
        os.environ['TESTING'] = os.path.join(
            storing, 'numerai_tournament_data.csv')
    if 'PREDICTING' not in os.environ:
        os.environ['PREDICTING'] = os.path.join(
            storing, 'predictions.csv')
    if 'PREPARED_TRAINING' not in os.environ:
        os.environ['PREPARED_TRAINING'] = os.path.join(
            storing, 'train_data.csv')
    if 'PREPARED_VALIDATING' not in os.environ:
        os.environ['PREPARED_VALIDATING'] = os.path.join(
            storing, 'valid_data.csv')
    if 'PREPARED_TESTING' not in os.environ:
        os.environ['PREPARED_TESTING'] = os.path.join(
            storing, 'test_data.csv')


def merge_tsne(selection):
    prefix = os.getenv('STORING')
    waiting = bool(int(os.getenv('WAITING', '0')))
    each = []
    each.append(os.path.join(prefix, 'tsne_2d_5p.npz'))
    each.append(os.path.join(prefix, 'tsne_2d_10p.npz'))
    each.append(os.path.join(prefix, 'tsne_2d_15p.npz'))
    each.append(os.path.join(prefix, 'tsne_2d_30p.npz'))
    each.append(os.path.join(prefix, 'tsne_2d_50p.npz'))
    each.append(os.path.join(prefix, 'tsne_2d_5p_poly.npz'))
    each.append(os.path.join(prefix, 'tsne_2d_10p_poly.npz'))
    each.append(os.path.join(prefix, 'tsne_2d_15p_poly.npz'))
    each.append(os.path.join(prefix, 'tsne_2d_30p_poly.npz'))
    each.append(os.path.join(prefix, 'tsne_2d_50p_poly.npz'))
    if not bool(int(os.getenv('TSNE_2D_ONLY', '0'))):
        each.append(os.path.join(prefix, 'tsne_3d_30p.npz'))
    while waiting:
        waiting = False
        for item in each:
            if not os.path.isfile(item):
                waiting = True
    selected = [np.load(each[i]) for i in selection]
    X_train = np.concatenate([item['train'] for item in selected], axis=1)
    X_valid = np.concatenate([item['valid'] for item in selected], axis=1)
    X_test = np.concatenate([item['test'] for item in selected], axis=1)
    np.savez(
        os.path.join(prefix, 'tsne.npz'),
        X_train=X_train, X_valid=X_valid, X_test=X_test)


def announce(text):
    sys.stdout.write('{}\n'.format('-' * 80))
    sys.stdout.write('{}\n'.format(text))
    sys.stdout.write('{}\n'.format('-' * 80))
    sys.stdout.flush()


def remember(suffix):
    predicting = os.getenv('PREDICTING')
    shutil.copyfile(predicting, predicting + suffix)


def main():
    operation = os.getenv('OPERATION', 'All')
    prepare()
    if operation in ['PrepareData', 'All']:
        announce('Data Preparation')
        os.system('python3 /code/prep_data.py')
    if operation in ['LogisticRegression', 'All']:
        announce('Simple Logistic Regression')
        os.system('python3 /code/models/pipeline/simple.py')
        remember('.simple')
    if operation in ['tSNE2D', 'All']:
        announce('t-SNE 2D')
        os.system('python3 /code/fit_tsne.py')
    if operation in ['tSNE3D', 'All'] and not bool(int(os.getenv('TSNE_2D_ONLY', '0'))):
        announce('t-SNE 3D')
        os.system('python3 /code/fit_tsne_3d.py')
    if operation in ['tSNESummary', 'All']:
        announce('t-SNE Summary')
        merge_tsne([1])
    if operation in ['TFNN', 'All']:
        announce('TF NN')
        os.system('python3 /code/models/classifier/main.py')
        remember('.tf_classifier')
    if operation in ['BasicVisualization', 'All']:
        announce('Basic data visualization notebook')
        os.system('python3 /code/notebooks/numerai.py')
    if operation in ['AdditionalVisualization', 'All']:
        announce('Additional data visualization notebook')
        os.system('python3 /code/notebooks/visualization.py')
    if operation in ['TFAutoencoder', 'All']:
        announce('TF Autoencoder')
        os.system('python3 /code/models/autoencoder/main.py')
    if operation in ['TFAdversarial', 'All']:
        announce('TF Adversarial')
        os.system('python3 /code/models/adversarial/main.py')
    if operation in ['TFPairwise', 'All']:
        announce('TF Pairwise')
        os.system('python3 /code/models/pairwise/main.py')
        remember('.tf_pairwise')
    if operation in ['Pairwise', 'All']:
        announce('Pairwise Interactions')
        os.system('python3 /code/models/pipeline/pairwise.py')
        remember('.pairwise')
    if operation in ['ParameterSearch', 'All']:
        announce('Searching parameters')
        os.system('python3 /code/search_params.py')
    if operation in ['AdditionalLogisticRegression', 'All']:
        announce('Logistic Regression')
        os.system('python3 /code/models/pipeline/lr.py')
        remember('.lr')
    if operation in ['FactorizationMachines', 'All']:
        announce('Factorization Machines')
        os.system('python3 /code/models/pipeline/fm.py')
        remember('.fm')
    if operation in ['GradientBoostingTrees', 'All']:
        announce('GBT')
        os.system('python3 /code/models/pipeline/gbt.py')
        remember('.gbt')
    if operation in ['Ensemble', 'All']:
        announce('Ensemble')
        os.system('python3 /code/ensemble.py')
    if operation in ['TPOT', 'All']:
        announce('TPOT')
        os.system('python3 /code/tpot_test.py')
        remember('.tpot')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


def main():
    base = os.path.dirname(os.path.realpath(__file__))
    operation = os.getenv('OPERATION')
    if operation == 'WideAndDeepCheck':
        os.system('python3 {}'.format(os.path.join(base, 'wide_n_deep_run.py')))
    if operation == 'WideAndDeep':
        os.system('python3 {}'.format(os.path.join(base, 'wide_n_deep_test_run.py')))
    if operation == 'XGBoostCheck':
        os.system('python3 {}'.format(os.path.join(base, 'xgboost_run.py')))
    if operation == 'XGBoost':
        os.system('python3 {}'.format(os.path.join(base, 'xgboost_test_predict.py')))


if __name__ == '__main__':
    main()

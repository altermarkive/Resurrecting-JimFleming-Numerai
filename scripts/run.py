#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


def main():
    base = os.path.dirname(os.path.realpath(__file__))
    operation = os.getenv('OPERATION')
    if operation == 'SVM':
        os.system('python3 {}'.format(os.path.join(base, 'svm.py')))
    if operation == 'RFC':
        os.system('python3 {}'.format(os.path.join(base, 'rfc.py')))
    if operation == 'RFCGrid':
        os.system('python3 {}'.format(os.path.join(base, 'rfc_grid_search.py')))


if __name__ == '__main__':
    main()

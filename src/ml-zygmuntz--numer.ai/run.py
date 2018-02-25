#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os


def main():
    base = os.path.dirname(os.path.realpath(__file__))
    operation = os.getenv('OPERATION')
    if operation == 'RFC':
        # os.system('python3 {}'.format(os.path.join(base, 'validate.py')))
        os.system('python3 {}'.format(os.path.join(base, 'predict.py')))
    if operation == 'LogisticRegression':
        if os.getenv('SELECTING') is None:
            if os.getenv('VALIDATING') is None:
                os.environ['VALIDATING'] = '/tmp/result.json'
            os.system('python3 {}'.format(os.path.join(base, 'validate_lr.py')))
            with open(os.getenv('VALIDATING'), 'rb') as handle:
                result = json.loads(handle.read().decode('utf-8'))
                selecting = int(result['selecting'])
                os.environ['SELECTING'] = str(selecting)
        os.system('python3 {}'.format(os.path.join(base, 'predict_lr.py')))
    if operation == 'ConsistencyCheck':
        os.system('python3 {}'.format(os.path.join(base, 'check_consistency.py')))


if __name__ == '__main__':
    main()

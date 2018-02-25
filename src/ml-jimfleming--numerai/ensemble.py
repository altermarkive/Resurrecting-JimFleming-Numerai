from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import glob
import time
import numpy as np
import os
import pandas as pd

paths = os.getenv('ENSEMBLING', None)
if None == paths:
    paths = [
        # glob.glob(os.getenv('PREDICTING') + '.simple')[0],
        glob.glob(os.getenv('PREDICTING') + '.lr')[0],
        glob.glob(os.getenv('PREDICTING') + '.fm')[0],
        # glob.glob(os.getenv('PREDICTING') + '.tf_pairwise')[0],
        # glob.glob(os.getenv('PREDICTING') + '.tf_classifier')[0],
        glob.glob(os.getenv('PREDICTING') + '.gbt')[0],
        glob.glob(os.getenv('PREDICTING') + '.pairwise')[0]
    ]
else:
    paths = paths.split(':')

def main():
    t_id = []
    probs = []
    for path in paths:
        df = pd.read_csv(path)
        t_id = df['id'].values
        probs.append(df['probability'].values)

    probability = np.power(np.prod(probs, axis=0), 1.0 / len(paths))
    assert(len(probability) == len(t_id))

    df_pred = pd.DataFrame({
        'id': t_id,
        'probability': probability,
    })
    csv_path = os.getenv('PREDICTING')
    df_pred.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))

if __name__ == '__main__':
    main()

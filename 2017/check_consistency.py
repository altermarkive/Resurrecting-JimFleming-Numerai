#!/usr/bin/env python3

"check validation consistency of predictions"

import json

import sys
import pandas as pd

from math import log
from sklearn.metrics import log_loss

import os

submission_file = os.getenv('PREDICTING')

test_file = os.getenv('TESTING')

try:
	print("loading {}...".format(submission_file))
	s = pd.read_csv(submission_file, header=0)
except:
	print("\nUsage: check_consistency.py <predictions file> <test file>")
	print("  i.e. check_consistency.py p.csv numerai_tournament_data.csv\n")
	raise SystemExit

print("loading {}...\n".format(test_file))
test = pd.read_csv(test_file, header=0)

v = test[ test.data_type == 'validation' ].copy()
v = v.merge( s, on = 'id', how = 'left' )

eras = v.era.unique()

good_eras = 0

results = {'eras': []}

for era in eras:
	tmp = v[ v.era == era ]
	ll = log_loss( tmp.target, tmp.probability )
	is_good = ll < -log( 0.5 )
	
	if is_good:
		good_eras += 1
	
	print("{} {} {:.2%} {}".format(era, len(tmp), ll, is_good))
	is_good = 'true' if is_good else 'false'
	result = {'era': era, 'count': len(tmp), 'log_loss': ll, 'ok': is_good}
	results['eras'].append(result)

consistency = good_eras / float( len( eras ))
print("\nconsistency: {:.1%} ({}/{})".format(consistency, good_eras, len(eras)))
results['consistency'] = consistency

ll = log_loss( v.target, v.probability )
print("log loss:    {:.2%}\n".format(ll))
results['log_loss'] = ll

with open(os.getenv('CHECKING'), 'wb') as handle:
    pretty = json.dumps(results, indent=2, separators=(',', ': '))
    handle.write(pretty.encode('utf-8'))

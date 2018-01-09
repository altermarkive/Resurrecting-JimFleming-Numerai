#!/bin/bash

set -e

if [ -z "$MEMORY_LIMIT" ]
then
  MEMORY_LIMIT=10240
fi
if [ -z "$TIME_LIMIT" ]
then
  TIME_LIMIT=60
fi
if [ -z "$PARALLEL_RUNS" ]
then
  PARALLEL_RUNS=1
fi

echo "Converting training data set..."
ROWS_TRAINING=$(cat $TRAINING | grep -c .)
echo " Training data set size: $ROWS_TRAINING"
su user -c "java -Xmx${MEMORY_LIMIT}m weka.core.converters.CSVLoader $TRAINING -B $ROWS_TRAINING > /tmp/training.all.arff"
su user -c "java -Xmx${MEMORY_LIMIT}m weka.filters.unsupervised.attribute.Remove -R 1-3 -i /tmp/training.all.arff -o /tmp/training.arff"
sed -i 's/target.*/class {0, 1}/g' /tmp/training.arff

echo "Converting tournament data set..."
ROWS_TESTING=$(cat $TESTING | grep -c .)
echo " Tournament data set size: $ROWS_TESTING"
su user -c "java -Xmx${MEMORY_LIMIT}m weka.core.converters.CSVLoader $TESTING -B $ROWS_TESTING > /tmp/testing.all.arff"
su user -c "java -Xmx${MEMORY_LIMIT}m weka.filters.unsupervised.attribute.Remove -R 1-3 -i /tmp/testing.all.arff -o /tmp/testing.arff"
sed -i 's/target.*/class {0, 1}/g' /tmp/testing.arff

echo "Training..."
su user -c "java -Xmx${MEMORY_LIMIT}m weka.classifiers.meta.AutoWEKAClassifier -t /tmp/training.arff -memLimit ${MEMORY_LIMIT} -timeLimit ${TIME_LIMIT} -parallelRuns ${PARALLEL_RUNS} -no-cv -d /tmp/auto-weka.model"

echo "Predicting..."
su user -c "java -Xmx${MEMORY_LIMIT}m weka.classifiers.meta.AutoWEKAClassifier -l /tmp/auto-weka.model -T /tmp/testing.arff -p 0" > ${PREDICTING}.raw

echo "Storing..."
cat $TESTING | tail -n +2 | awk -F',' '{ print $1 }' > /tmp/ids.csv
cat ${PREDICTING}.raw \
| grep -v '^$' \
| tail -n +3 \
| awk -F':' '{ print $3 }' \
| sed 's/\+/\ /g' \
| awk '{ if ($1 == 1) { print $2 } else { print 1.0 - $2 } }' \
| awk '{ if ($1 == 1) { print "0.9999999999999998" } else { if ($1 == 0) { print "0.0000000000000002" } else { print $1 } } }' > /tmp/predictions.csv
echo "id,probability" > $PREDICTING
paste -d, /tmp/ids.csv /tmp/predictions.csv >> $PREDICTING

echo "Done."

#!/usr/bin/env zsh

for d in /data1/centrioles/* ; do
  python ../src/centrack/experiments/train_test.py "$d"  1 2 3;
done

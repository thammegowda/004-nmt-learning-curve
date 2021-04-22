#!/usr/bin/env bash

file=learning-curve-data.tgz
dir=learning-curve-data
[[ -f $file ]] ||  wget http://www.statmt.org/challenges/learning-curve-data.tgz
[[ -d $dir ]] || tar xvf $dir

#mtdata get -l end-spa -ts newstest201{2,3} -o data
#for i in data/tests/newstest*; do mv $i ${i/-eng_spa}; done

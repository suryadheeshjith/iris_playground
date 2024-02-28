#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local train
# ./.python-greene submitit_hydra.py $comp exp=trainer name="$(date +%F)-trainer_test"

# local eval
./.python-greene submitit_hydra.py $comp exp=eval name="$(date +%F)-eval_test"
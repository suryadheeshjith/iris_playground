#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# test mae crop
./.python-greene submitit_hydra.py $comp exp=trainer name="$(date +%F)-trainer_test"
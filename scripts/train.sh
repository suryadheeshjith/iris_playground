#!/bin/bash


./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=trainer name="$(date +%F)-1GPU_train_state_assault"

#!/bin/bash


./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000 exp=trainer name="$(date +%F)-1GPU_train_img_breakout"

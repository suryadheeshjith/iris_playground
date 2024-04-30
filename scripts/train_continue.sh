#!/bin/bash


###### 
# Image Space
###### 


### Breakout
# img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_12hrs exp=trainer_img name="2024-04-28-img_breakout" env.train.id=BreakoutNoFrameskip-v4 training.load_bc_agent=False common.resume=True continue_dir="/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-28-img_breakout/imgbreakout" resume_id="amebwzt9"

# Naive BCRL img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_12hrs exp=trainer_img_naive_bcrl name="2024-04-28-img_breakout_naivebcrl" env.train.id=BreakoutNoFrameskip-v4 training.load_bc_agent=False common.resume=True continue_dir="/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-28-img_breakout_naivebcrl/naive" resume_id="zrs2ibsh"

# Regularized Loaded BC img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="2024-04-28-img_breakout_regbcrl_exp" training.regularize_schedule=exp bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-27-breakout_150k_21stacked/saved_npy" env.train.id=BreakoutNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4 datasets.train.max_ram_usage=170G training.load_bc_agent=False common.resume=True continue_dir="/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-28-img_breakout_regbcrl_exp/exp" resume_id="1uyhvsik"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="2024-04-28-img_breakout_regbcrl_linear" training.regularize_schedule=linear bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-27-breakout_150k_21stacked/saved_npy" env.train.id=BreakoutNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4 datasets.train.max_ram_usage=170G training.load_bc_agent=False common.resume=True continue_dir="/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-28-img_breakout_regbcrl_linear/linear" resume_id="7xepv4a7"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="2024-04-28-img_breakout_regbcrl_cos" training.regularize_schedule=cosine bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-27-breakout_150k_21stacked/saved_npy" env.train.id=BreakoutNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4 datasets.train.max_ram_usage=170G training.load_bc_agent=False common.resume=True continue_dir="/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-28-img_breakout_regbcrl_cos/cos" resume_id="zl2f4lu8"
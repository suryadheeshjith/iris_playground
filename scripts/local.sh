#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local train

###### 
# Image Space
###### 


### Breakout
# img train
# ./.python-greene submitit_hydra.py $comp exp=trainer_img wandb.mode=disabled name="$(date +%F)-trainer_test_img_breakout" env.train.id=BreakoutNoFrameskip-v4

# BC train
# ./.python-greene submitit_hydra.py $comp exp=trainer_img_bc wandb.mode=disabled name="$(date +%F)-trainer_test_img_breakout" bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-27-breakout_150k_21stacked/saved_npy" env.train.id=BreakoutNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4

# Naive BCRL img train
# ./.python-greene submitit_hydra.py $comp exp=trainer_img_naive_bcrl training.actor_critic.start_after_epochs=1 wandb.mode=disabled name="$(date +%F)-trainer_test_img_breakout_naivebcrl" initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/.LOCAL/trainer/2024-04-27-trainer_test_img_breakout_bc_train_100k_21stacked_wd/bc_checkpoints/last.pt' env.train.id=BreakoutNoFrameskip-v4

# Regularized Loaded BC img train
# ./.python-greene submitit_hydra.py $comp exp=trainer_img_regularized_bcrl training.actor_critic.start_after_epochs=1 wandb.mode=disabled name="$(date +%F)-trainer_test_img_breakout_regbcrl" training.regularize_schedule=exp initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/.LOCAL/trainer/2024-04-27-trainer_test_img_breakout_bc_train_100k_21stacked_wd/bc_checkpoints/last.pt' env.train.id=BreakoutNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4 bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-27-breakout_150k_21stacked/saved_npy" datasets.train.max_ram_usage=170G


# Train continue
./.python-greene submitit_hydra.py $comp exp=trainer_img name="$(date +%F)-img_breakout" env.train.id=BreakoutNoFrameskip-v4 training.load_bc_agent=False common.resume=True continue_dir="/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-28-img_breakout/imgbreakout"

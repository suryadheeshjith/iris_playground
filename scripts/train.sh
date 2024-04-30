#!/bin/bash


###### 
# Image Space
###### 

###
# BC Train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_bc name="$(date +%F)-img_breakout_bc_train" bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-27-breakout_150k_21stacked/saved_npy" env.train.id=BreakoutNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_bc name="$(date +%F)-img_mspacman_bc_train" bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-29-pacman_150k_21stacked/saved_npy" env.train.id=MsPacmanNoFrameskip-v4 datasets.bc.env_name=MsPacmanNoFrameskip-v4

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_bc name="$(date +%F)-img_demonattack_bc_train" bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-29-demonattack_150k_21stacked/saved_npy" env.train.id=DemonAttackNoFrameskip-v4 datasets.bc.env_name=DemonAttackNoFrameskip-v4


### Breakout
# img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img  name="$(date +%F)-img_breakout" env.train.id=BreakoutNoFrameskip-v4

# Naive BCRL img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_naive_bcrl name="$(date +%F)-img_breakout_naivebcrl" initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-28-img_breakout_bc_train/bc/bc_checkpoints/last.pt' env.train.id=BreakoutNoFrameskip-v4

# Regularized Loaded BC img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="$(date +%F)-img_breakout_regbcrl_cos" training.regularize_schedule=cosine initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-28-img_breakout_bc_train/bc/bc_checkpoints/last.pt' bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-27-breakout_150k_21stacked/saved_npy" env.train.id=BreakoutNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4 datasets.train.max_ram_usage=170G

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="$(date +%F)-img_breakout_regbcrl_exp" training.regularize_schedule=exp initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-28-img_breakout_bc_train/bc/bc_checkpoints/last.pt' bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-27-breakout_150k_21stacked/saved_npy" env.train.id=BreakoutNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4 datasets.train.max_ram_usage=170G

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="$(date +%F)-img_breakout_regbcrl_constant" training.regularize_schedule=constant initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-28-img_breakout_bc_train/bc/bc_checkpoints/last.pt' bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-27-breakout_150k_21stacked/saved_npy" env.train.id=BreakoutNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4 datasets.train.max_ram_usage=170G



### MS Pacman
# img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img  name="$(date +%F)-img_mspacman" env.train.id=MsPacmanNoFrameskip-v4

# Naive BCRL img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_naive_bcrl name="$(date +%F)-img_mspacman_naivebcrl" initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-29-img_mspacman_bc_train/mspacman/bc_checkpoints/93.pt' env.train.id=MsPacmanNoFrameskip-v4

# Regularized Loaded BC img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="$(date +%F)-img_mspacman_regbcrl_cos" training.regularize_schedule=cosine initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-29-img_mspacman_bc_train/mspacman/bc_checkpoints/93.pt' bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-29-pacman_150k_21stacked/saved_npy" env.train.id=MsPacmanNoFrameskip-v4 datasets.bc.env_name=MsPacmanNoFrameskip-v4 datasets.train.max_ram_usage=175G

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="$(date +%F)-img_mspacman_regbcrl_exp" training.regularize_schedule=exp initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-29-img_mspacman_bc_train/mspacman/bc_checkpoints/93.pt' bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-29-pacman_150k_21stacked/saved_npy" env.train.id=MsPacmanNoFrameskip-v4 datasets.bc.env_name=MsPacmanNoFrameskip-v4 datasets.train.max_ram_usage=175G

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="$(date +%F)-img_mspacman_regbcrl_constant" training.regularize_schedule=constant initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-29-img_mspacman_bc_train/mspacman/bc_checkpoints/93.pt' bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-29-pacman_150k_21stacked/saved_npy" env.train.id=MsPacmanNoFrameskip-v4 datasets.bc.env_name=MsPacmanNoFrameskip-v4 datasets.train.max_ram_usage=175G


### Demon Attack
# img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img name="$(date +%F)-img_demonattack" env.train.id=DemonAttackNoFrameskip-v4

# Naive BCRL img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_naive_bcrl name="$(date +%F)-img_demonattack_naivebcrl" initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-29-img_demonattack_bc_train/demonattack/bc_checkpoints/88.pt' env.train.id=DemonAttackNoFrameskip-v4

# Regularized Loaded BC img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="$(date +%F)-img_demonattack_regbcrl_cos" training.regularize_schedule=cosine initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-29-img_demonattack_bc_train/demonattack/bc_checkpoints/88.pt' bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-29-demonattack_150k_21stacked/saved_npy" env.train.id=DemonAttackNoFrameskip-v4 datasets.bc.env_name=DemonAttackNoFrameskip-v4 datasets.train.max_ram_usage=173G

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="$(date +%F)-img_demonattack_regbcrl_exp" training.regularize_schedule=exp initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-29-img_demonattack_bc_train/demonattack/bc_checkpoints/88.pt' bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-29-demonattack_150k_21stacked/saved_npy" env.train.id=DemonAttackNoFrameskip-v4 datasets.bc.env_name=DemonAttackNoFrameskip-v4 datasets.train.max_ram_usage=173G

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_img_regularized_bcrl name="$(date +%F)-img_demonattack_regbcrl_constant" training.regularize_schedule=constant initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/trainer/2024-04-29-img_demonattack_bc_train/demonattack/bc_checkpoints/88.pt' bc_datapath="/scratch/lvb243/ddrl-project/Video-Diffusion-Models/.LOCAL/save_with_atariari/2024-04-29-demonattack_150k_21stacked/saved_npy" env.train.id=DemonAttackNoFrameskip-v4 datasets.bc.env_name=DemonAttackNoFrameskip-v4 datasets.train.max_ram_usage=173G

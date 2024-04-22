#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local train
### Breakout
# State space train
# ./.python-greene submitit_hydra.py $comp exp=trainer name="$(date +%F)-trainer_test_breakout_bc_loaded" env.train.id=Breakout-ramNoFrameskip-v4 wandb.mode=disabled

# BC train
# ./.python-greene submitit_hydra.py $comp exp=trainer name="$(date +%F)-trainer_test_breakout_bc_loaded" bc.should=True env.train.id=Breakout-ramNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4 wandb.mode=disabled

# BC initialized state space train
# ./.python-greene submitit_hydra.py $comp exp=trainer name="$(date +%F)-trainer_test_breakout_bc_loaded" training.load_bc_agent=True initialization.path_to_checkpoint='/scratch/lvb243/ddrl-project/iris_playground/.LOCAL/trainer/2024-04-22-trainer_test_breakout_bc/bc_checkpoints/last.pt' initialization.load_actor_critic=True env.train.id=Breakout-ramNoFrameskip-v4 wandb.mode=disabled 

### Space Invaders
# State space train
# ./.python-greene submitit_hydra.py $comp exp=trainer name="$(date +%F)-trainer_test_breakout_bc_loaded" env.train.id=SpaceInvaders-ramNoFrameskip-v4 wandb.mode=disabled


### Ms Pacman
# State space train
./.python-greene submitit_hydra.py $comp exp=trainer name="$(date +%F)-trainer_test_breakout_bc_loaded" env.train.id=MsPacman-ramNoFrameskip-v4 wandb.mode=disabled


# local eval - will not work
# ./.python-greene submitit_hydra.py $comp exp=eval name="$(date +%F)-eval_test"
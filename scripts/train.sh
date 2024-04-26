#!/bin/bash


###### 
# Image Space
###### 


### Breakout
# img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=trainer_img name="$(date +%F)-img_breakout" env.train.id=BreakoutNoFrameskip-v4

# BC train
./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_3hrs exp=trainer_img name="$(date +%F)-img_breakout_bc_train" bc.should=True env.train.id=BreakoutNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4

# BC initialized img train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=trainer_img name="$(date +%F)-img_breakout_bc_loaded" training.load_bc_agent=True initialization.path_to_checkpoint='' initialization.load_actor_critic=True env.train.id=BreakoutNoFrameskip-v4 


###### 
# State Space
###### 

### Breakout
# Train State space model
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=trainer name="$(date +%F)-train_ss_breakout" env.train.id=Breakout-ramNoFrameskip-v4

# Train BC (Make sure the dataset is present in /scratch/lvb243/ddrl-project/bc_generator/trajectories)
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=trainer name="$(date +%F)-train_bc_breakout" bc.should=True env.train.id=Breakout-ramNoFrameskip-v4 datasets.bc.env_name=BreakoutNoFrameskip-v4

# Train BC-initialized State space model
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=trainer name="$(date +%F)-train_state_space_breakout" training.load_bc_agent=True initialization.path_to_checkpoint='' initialization.load_actor_critic=True env.train.id=Breakout-ramNoFrameskip-v4


### Space Invaders
# State space train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=trainer name="$(date +%F)-train_ss_spaceinvaders" env.train.id=SpaceInvaders-ramNoFrameskip-v4

### Ms Pacman
# State space train
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=trainer name="$(date +%F)-train_ss_mspacman" env.train.id=MsPacman-ramNoFrameskip-v4


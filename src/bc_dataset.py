import os

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from PIL import Image
from pathlib import Path

def get_all_files(directory):
    files = [f for f in os.listdir(directory) if
             os.path.isfile(os.path.join(directory, f))]
    return files

class BCDataset(Dataset):
    def __init__(self, main_folder, env_name, train = True, is_ram = False, name = None):
        self.main_folder = main_folder
        self.env_name = env_name
        self.name = name if name is not None else 'dataset'
        self.states = []
        self.targets = []
        if train:
            self.data_path = Path(self.main_folder) / self.env_name / 'train'
        else:
            self.data_path = Path(self.main_folder) / self.env_name / 'val'
        self.is_ram = is_ram
        print("Loading Data...")
        self._load_data()
        
    def _load_data(self):
        data_files = get_all_files(self.data_path)
        num_eps = len(data_files) // 2
        action_list = []
        state_list = []

        for i in range(1,num_eps+1):
            actions, states = self._load_state_action_pair_traj(
                i, stateskip=1, offset=0)
            action_list.extend(actions)
            state_list.extend(states)
    
        
        assert len(state_list) == len(action_list)
        print("Loaded {0} samples".format(len(state_list)))
        self.states = state_list # Each state is of shape [1, stacksize, 128]
        self.targets = action_list
    
    def __len__(self):
        return len(self.states)

    def sample_batch(self, batch_num_samples: int):
        indices = random.sample(range(len(self.states)), batch_num_samples)
        states = torch.stack([torch.tensor(self.states[i]) for i in indices])
        targets = torch.stack([torch.tensor(self.targets[i]) for i in indices])
        batch = {'observations': states / 255.0, 'actions': targets}    
        return batch
    
    def _load_state_action_pair_traj(self, ind, stateskip=1, offset=0):

        assert stateskip == 1 and offset == 0    
        frames_path = self.data_path / "{0}_frames.npy".format(ind)
        actions_path = self.data_path / "{0}_actions.npy".format(ind)
        print("Loading {0}".format(frames_path))
        frames = np.load(frames_path)
        print("Loading {0}".format(actions_path))
        try:
            actions = np.load(actions_path)
        except ValueError:
            print("Could not load {0}".format(actions_path)) # No idea why this happens
            return [], []

        state_list = list(frames)
        action_list = list(actions)

        return action_list, state_list


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    main_folder = '/scratch/lvb243/ddrl-project/Video-Diffusion-Models/saved_npy'
    env_name = 'BreakoutNoFrameskip-v4'

    train_dataset = BCDataset(main_folder, env_name)
    batch = train_dataset.sample_batch(32)

    print(batch['observations'].shape)
    print(batch['actions'].shape)
    
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

def get_all_files(directory):
    files = [f for f in os.listdir(directory) if
             os.path.isfile(os.path.join(directory, f))]
    return files

class BCDataset(Dataset):
    def __init__(self, main_folder, env_name, train = True, is_ram = False, is_full=False, name = None):
        self.main_folder = main_folder
        self.env_name = env_name
        self.name = name if name is not None else 'dataset'
        self.is_full = is_full
        self.states = []
        self.targets = []
        self.is_ram = is_ram
        if train:
            self.data_path = Path(self.main_folder) / self.env_name / 'train'
            print("Loading Train Data...")
        else:
            self.data_path = Path(self.main_folder) / self.env_name / 'val'
            print("Loading Val Data...")
        self._load_data()
        
    def _load_data(self):
        if self.is_full:
            state_list = np.load(self.data_path / '0_frames.npy', allow_pickle=True)
            action_list = np.load(self.data_path / '0_actions.npy', allow_pickle=True)
        else:
            data_files = get_all_files(self.data_path)
            num_eps = len(data_files) // 2
            action_list = []
            state_list = []

            for i in tqdm(range(1,num_eps+1)):
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
    
    def random_sample_batch(self, batch_size):
        indices = random.sample(range(len(self.states)), batch_size)
        states = torch.stack([self.augment(torch.tensor(self.states[i])) for i in indices])
        targets = torch.stack([torch.tensor(self.targets[i]) for i in indices])
        batch = {'observations': states / 255.0, 'actions': targets}    
        return batch

    def start_epoch(self, batch_size):
        size = len(self.states) // batch_size * batch_size
        ls = torch.randperm(size)
        self.random_indices_list = ls.reshape(-1, batch_size)
        assert self.random_indices_list.shape[0] == len(self.states) // batch_size

    def sample_batch(self, idx: int):
        indices = self.random_indices_list[idx]
        states = torch.stack([self.augment(torch.tensor(self.states[i])) for i in indices])
        targets = torch.stack([torch.tensor(self.targets[i]) for i in indices])
        batch = {'observations': states / 255.0, 'actions': targets}    
        return batch
    
    def _load_state_action_pair_traj(self, ind, stateskip=1, offset=0):

        assert stateskip == 1 and offset == 0    
        frames_path = self.data_path / "{0}_frames.npy".format(ind)
        actions_path = self.data_path / "{0}_actions.npy".format(ind)
        # print("Loading {0}".format(frames_path))
        frames = np.load(frames_path)
        # print("Loading {0}".format(actions_path))
        try:
            actions = np.load(actions_path)
        except ValueError:
            # print("Could not load {0}".format(actions_path)) # No idea why this happens
            return [], []

        state_list = list(frames)
        action_list = list(actions)

        return action_list, state_list

    def augment(self, x, pad=4):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    main_folder = '/scratch/lvb243/ddrl-project/Video-Diffusion-Models/saved_npy'
    env_name = 'BreakoutNoFrameskip-v4'

    train_dataset = BCDataset(main_folder, env_name)
    batch = train_dataset.sample_batch(32)

    print(batch['observations'].shape)
    print(batch['actions'].shape)
    
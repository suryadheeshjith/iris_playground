import os

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from einops import rearrange
from PIL import Image
from torchvision.transforms import Resize

# NOTE: To get data with different sized stacks, you would have to generate it first.

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
        self.train = train
        self.is_ram = is_ram
        if not self.is_ram:
            self.resize_fn = Resize([64,64])
        self._load_data()
        
    def _load_data(self):
        data_path = os.path.join(self.main_folder, 'rand_steps=0', self.env_name)
        data_path_75 = os.path.join(self.main_folder, 'rand_steps=75', self.env_name)
        data_path_100 = os.path.join(self.main_folder, 'rand_steps=100', self.env_name)

        if self.train:
            min_traj=[0, 10000, 20000]
            max_traj=[159, 10039, 20039]
        else:
            min_traj=[160, 10040, 20040]
            max_traj=[199, 10049, 20049]
        
        folder_paths = [data_path, data_path_75, data_path_100]

        state_list = []
        actions_one_hot = []
        for i in range(len(folder_paths)):
            print("Processing folder ", folder_paths[i])
            states, actions = self._gather_states_and_actions(
                folder_paths[i], min_traj[i], max_traj[i], data_ext='npz'
            )
            if not self.is_ram:
                new_states = []
                for state in states:
                    temp = torch.from_numpy(state)
                    temp = rearrange(temp, 'C H W S -> S C H W')
                    temp = self.resize_fn(temp)
                    new_states.append(temp)
                states = new_states
                
            state_list.extend(states)
            actions_one_hot.extend(actions)

        self.states = state_list # Each state is of shape [1, stacksize, 128]
        self.targets = actions_one_hot
    
    def __len__(self):
        return len(self.states)

    def resize(self, obs):
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)

    def sample_batch(self, batch_num_samples: int):
        indices = random.sample(range(len(self.states)), batch_num_samples)
        states = torch.stack([self.states[i] for i in indices])
        targets = torch.tensor([self.targets[i] for i in indices])
        batch = {'observations': states / 255.0, 'actions': targets}    
        return batch
    
    def _load_state_action_pair_traj(self, trajectory_file, stateskip=1, offset=0):
        """
        Same as above, but for generated trajectory data.
        """
        data = np.load(trajectory_file)
        if offset > 0:
            state_list = data['obs'][::stateskip][:-offset]
        else:
            state_list = data['obs'][::stateskip]

        # In new versions of the generated trajectories, there is a column for
        # model selected actions, which we want to predict. This is to distinguish
        # between the chosen actions, which may be a sticky action.
        if 'actions' in data.keys():
            label = 'actions'
        elif 'model selected actions' in data.keys():
            label = 'model selected actions'
        action_list = data[label][:, 0][::stateskip][offset:]
        # Offset = 0 seems to be the correct amount of offset to make actions
        # and states line up correctly.

        return action_list, state_list


    def _gather_states_and_actions(self, main_folder, min_traj, max_traj, data_ext='npz'):
        data_files = [os.path.join(main_folder, f) for f in
                    get_all_files(main_folder)]
        action_list = []
        state_list = []

        def get_trajectory_num(x):
            assert x.split('.')[-1] == data_ext
            return int(x.split('_')[-1][: -(len(data_ext)+1)])

        data_files = [path for path in data_files if
                    min_traj <= get_trajectory_num(path) <= max_traj]
        for file in data_files:
            actions, states = self._load_state_action_pair_traj(
                file, stateskip=1, offset=0)
            action_list.extend(actions)
            state_list.extend(states)
        return state_list, action_list


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    main_folder = 'trajectories'
    env_name = 'BreakoutNoFrameskip-v4'

    train_dataset = BCDataset(main_folder, env_name)
    batch = train_dataset.sample_batch(32)

    print(batch['observations'].shape)
    print(batch['actions'].shape)
    
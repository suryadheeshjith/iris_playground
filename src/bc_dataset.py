import os

from torch.utils.data import Dataset
import numpy as np
import os

# NOTE: To get data with different sized stacks, you would have to generate it first.

def get_all_files(directory):
    files = [f for f in os.listdir(directory) if
             os.path.isfile(os.path.join(directory, f))]
    return files

class BCData(Dataset):
    def __init__(self, states, targets):
        self.states = states
        self.targets = targets
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):
        state = self.states[index]
        target = self.targets[index]
        return state, target
    
def load_state_action_pair_traj(trajectory_file, stateskip=1, offset=0):
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


def _gather_states_and_actions(main_folder, min_traj, max_traj, data_ext='npz'):
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
        actions, states = load_state_action_pair_traj(
            file, stateskip=1, offset=0)
        action_list.extend(actions)
        state_list.extend(states)
    return state_list, action_list

def get_dataset(main_folder, min_traj, max_traj, data_ext='npz'):
    if isinstance(main_folder, list):
        state_list = []
        actions_one_hot = []
        for i in range(len(main_folder)):
            states, actions = _gather_states_and_actions(
                main_folder[i], min_traj[i], max_traj[i], data_ext=data_ext
            )
            state_list.extend(states)
            actions_one_hot.extend(actions)
    else:
        state_list, actions_one_hot = _gather_states_and_actions(
            main_folder, min_traj, max_traj, data_ext=data_ext
        )

    data_gen = BCData(
        states=state_list,
        targets=actions_one_hot,
    )
    return data_gen


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    main_folder = 'trajectories'
    env_name = 'BreakoutNoFrameskip-v4'

    data_path = os.path.join(main_folder, 'rand_steps=0', env_name)
    data_path_75 = os.path.join(main_folder, 'rand_steps=75', env_name)
    data_path_100 = os.path.join(main_folder, 'rand_steps=100', env_name)

    train_dataset = get_dataset(
        [data_path, data_path_75, data_path_100],
        min_traj=[0, 10000, 20000],
        max_traj=[159, 10039, 20039]
    )

    data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for i, (states, target) in enumerate(data_loader):
        print(states.shape)
        print(target)
        if i > 2:
            break
    print('done')
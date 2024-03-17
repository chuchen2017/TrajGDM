import json
import torch
import random
from torch.utils.data import Dataset  # , Dataloader

class TrajectoryDataset(Dataset):
    def __init__(self, coder_data):
        self.data = []
        for traj in coder_data:
            x = torch.tensor(traj)  #, dtype=torch.long
            self.data.append(x)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def load_trajs(dataset='TDrive',batch_size=8,num_workers=4,train=True):
    if train:
        with open('../datasets/' + dataset + '/' + dataset + '_Train.json', 'r') as file:
            trajs = json.loads(file.read())
    else:
        with open('../datasets/' + dataset + '/' + dataset + '_Eval.json', 'r') as file:
            trajs = json.loads(file.read())
    trajs=trajs
    random.shuffle(trajs)
    dataset = TrajectoryDataset(coder_data=trajs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return dataloader

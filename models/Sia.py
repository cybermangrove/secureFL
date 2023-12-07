import numpy as np
import torch
import sys
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)#idxs already a python list

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class SIA(object):
    def __init__(self, args, w_locals=None, dataset=None, dict_sia_users=None):
        self.args = args
        self.w_locals = w_locals
        self.dataset = dataset
        self.dict_sia_users = dict_sia_users

    def attack(self, net):
        correct_total = 0
        len_set = 0
        for idx in self.dict_sia_users:
            #idx=0,1,...,9            
            dataset_local = DataLoader(DatasetSplit(self.dataset, self.dict_sia_users[idx]),
                                       batch_size=self.args.local_bs, shuffle=False)
            y_loss_all = []
            # evaluate the selected training data on each local model
            for local in self.dict_sia_users:
                y_loss_party = []
                idx_tensor = torch.tensor(idx)
                net.load_state_dict(self.w_locals[local])
                net.eval()
                for id, (data, target) in enumerate(dataset_local):
                    if self.args.gpu != -1:
                        data, target = data.to('cpu'), target.to('cpu') # data.cuda(), target.cuda()
                        idx_tensor = idx_tensor.to('cpu') # idx_tensor.cuda()
                    log_prob = net(data)
                    # prediction loss based attack: get the prediction loss of the target training sample
                    loss = nn.CrossEntropyLoss(reduction='none')
                    y_loss = loss(log_prob, target)
                    y_loss_party.append(y_loss.cpu().detach().numpy()) # turn torch into numpy array
                y_loss_party = np.concatenate(y_loss_party).reshape(-1)
                y_loss_all.append(y_loss_party)

            y_loss_all = np.array(y_loss_all)
            y_loss_all = torch.from_numpy(y_loss_all).to(self.args.device)

            index_of_min_loss = y_loss_all.min(0, keepdim=True)[1] # [1] return the indexes
            correct_local = index_of_min_loss.eq(
                idx_tensor.repeat_interleave(len(dataset_local.dataset))).long().cpu().sum()
            correct_total += correct_local
            len_set += len(dataset_local.dataset)

        # calculate source inference attack accuracy
        accuracy_sia = 100.00 * correct_total / len_set
        # print('Prediction loss based source inference attack accuracy: {}/{} ({:.2f}%)'.format(correct_total, len_set, accuracy_sia))
        return accuracy_sia

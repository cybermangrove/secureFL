import numpy as np
import torch
import copy
import random
import sys
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models.Nets import model_dict_to_list, list_to_model_dict

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)#idxs already a python list

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class SimulationAttack(object):
    def __init__(self, args, model_dict=None, dataset=None, dict_simulation_users=None):
        self.args = args
        self.model_dict = model_dict
        self.dataset = dataset
        self.dict_simulation_users = dict_simulation_users

    def attack(self, net):
        #server
        all_parameters = model_dict_to_list(self.model_dict)
        result_encrypted_index = []
        
        if self.args.encrypt_percent == 0:
            result_encrypted_index = []
        elif self.args.encrypt_percent == 1:
            result_encrypted_index = list(range(len(all_parameters)))
        else:
            y_loss_choices = []
            all_encrypted_index = []
            choices = self.args.server_choices
            for choice in range(choices):
                encrypted_index = np.random.choice(len(all_parameters), int(self.args.encrypt_percent * len(all_parameters)), replace=False)
                # print(f'choice{choice}: size{len(encrypted_index)}, detail: {encrypted_index}')
                all_encrypted_index.append(encrypted_index)
                print('.', end='')
                y_loss_choice_sum = 0
                #encrypted_index发送给clients
                for idx in self.dict_simulation_users:
                    #idx=0,1,...,9
                    idx_tensor = torch.tensor(idx)
                    dataset_local = DataLoader(DatasetSplit(self.dataset, self.dict_simulation_users[idx]), batch_size=self.args.local_bs, shuffle=False)

                    decorated_parameters = []
                    encrypted_parameters = []
                    non_encrypted_parameters = []
                    for i in range(len(all_parameters)):
                        if i in encrypted_index:
                            encrypted_parameters.append(all_parameters[i])
                            decorated_parameters.append(random.uniform(-2, 2)) # random -2~-2
                        else:
                            non_encrypted_parameters.append(all_parameters[i])
                            decorated_parameters.append(all_parameters[i])
                    new_model_dict = copy.deepcopy(self.model_dict)
                    new_model_dict = list_to_model_dict(new_model_dict, decorated_parameters)
                    net.load_state_dict(new_model_dict)
                    net.eval()
                    y_loss_party = []
                    for id, (data, target) in enumerate(dataset_local):
                        if self.args.gpu != -1:
                            data, target = data.to('cpu'), target.to('cpu') # data.cuda(), target.cuda()
                            idx_tensor = idx_tensor.to('cpu') # idx_tensor.cuda()
                        log_prob = net(data)
                        loss = nn.CrossEntropyLoss(reduction='none')
                        y_loss = loss(log_prob, target)
                        y_loss_party.append(y_loss.cpu().detach().numpy())
                    y_loss_party = np.concatenate(y_loss_party).reshape(-1)
                    y_loss_choice_sum += np.sum(y_loss_party)
                y_loss_choices.append(y_loss_choice_sum)
            print('')
            
            max_index = y_loss_choices.index(max(y_loss_choices))
            result_encrypted_index = all_encrypted_index[max_index]
            result_encrypted_index.sort()
        
        return result_encrypted_index
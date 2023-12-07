import torch
from torch import nn
import torch.nn.functional as F
import copy

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class Mnistcnn(nn.Module):
    def __init__(self, args):
        super(Mnistcnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64*4*4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, args.num_classes)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def model_dict_to_list(model_dict):
    all_parameters = []
    for _, value in model_dict.items():
        if isinstance(value, torch.Tensor):
            all_parameters.extend(value.view(-1).tolist())
    return all_parameters


def list_to_model_dict(model_dict, plain_list):
    new_model_dict = copy.deepcopy(model_dict)

    param_index = 0
    for key, value in new_model_dict.items():
        if isinstance(value, torch.Tensor):
            shape = value.shape
            new_value = torch.tensor(plain_list[param_index:param_index+value.numel()])
            new_model_dict[key] = new_value.view(shape)
            param_index += value.numel()
    
    return new_model_dict
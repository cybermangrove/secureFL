import torch.nn.functional as F
import math
from torch.utils.data import DataLoader


def test_fun(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            # data, target = data.cuda(), target.cuda()
            data, target = data.to('cpu'), target.to('cpu')
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

import math

def calculate_standard_deviation(numbers):
    if len(numbers) < 2:
        raise ValueError("At least two numbers are required")

    sum_numbers = sum(numbers)
    mean = sum_numbers / len(numbers)

    squared_differences = sum((num - mean) ** 2 for num in numbers)

    variance = squared_differences / (len(numbers) - 1)
    standard_deviation = math.sqrt(variance)

    return standard_deviation


def averaged_test_fun(net_g, datatest, args):
    acc_list = []
    max_loop_number = 2000
    for i in range(max_loop_number):
        acc, loss = test_fun(net_g, datatest, args)
        acc_list.append(acc)
        if len(acc_list) > 1:
            std = calculate_standard_deviation(acc_list)
            standard_error = std * 1.0 / math.sqrt(len(acc_list))
            if standard_error <= 0.01 and i >=100:
                # print(f'break averaged_test_fun_i:{i}')
                break
            # print(f'len(acc_list):{len(acc_list)}')
        i = i + 1
        # print(acc_list) #all elements are the same
    return sum(acc_list) / len(acc_list)
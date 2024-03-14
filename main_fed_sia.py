import copy
import numpy as np
import torch
import sys
import time

from models.Fed import FedAvg
from models.Nets import MLP, Mnistcnn
from models.Sia import SIA
from models.Update import LocalUpdate
from models.test import test_fun, averaged_test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser

def process(args):
    # load dataset and split data for users
    dataset_train, dataset_test, dict_party_user, dict_sample_user, dict_simulation_user = get_dataset(args)

    #data is X dimension=60 classified into Y whose classes=10
    #dict_party_user is the idx, randomly seperating dataset_train into {num_users}
    #dict_sample_user is the idx, with each element sized 100 and randomly picked from each dict_party_user element

    # build model
    if args.model == 'cnn' and args.dataset == 'MNIST':
        net_glob = Mnistcnn(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        dataset_train = dataset_train.dataset
        dataset_test = dataset_test.dataset
        img_size = dataset_train[0][0].shape
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    empty_net = net_glob
    print('Model architecture:')
    print(net_glob)
    net_glob.train()#Set the neural network model net _ glob to enter the training mode

    # copy weights
    w_glob = net_glob.state_dict()

    # initial settings
    total_time = 0
    total_acc_train = 0
    total_acc_test = 0
    total_attack_success_rate = 0
    total_loss_list = [0] * args.epochs

    duplication = 100
    for i in range(duplication):
        execution_time = 0
        acc_train = 0
        acc_test = 0
        # training
        if args.all_clients:
            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(args.num_users)]

        best_att_acc = 0
        att_acc_list = []
        for iter in range(args.epochs):
            print('+++')

            loss_locals = []
            if not args.all_clients:
                w_locals = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            print(f'Epoch Round {iter} Start, local train')
            start_time = time.time()
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx])

                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            
            #record time
            end_time = time.time()
            execution_time += end_time - start_time

            # implement the source inference attack
            SIA_attack = SIA(args=args, w_locals=w_locals, dataset=dataset_train, dict_sia_users=dict_sample_user)
            attack_acc = SIA_attack.attack(net=empty_net.to('cpu'))#args.device
            att_acc_list.append(attack_acc)
            best_att_acc = max(best_att_acc, attack_acc)

            start_time = time.time()
            #<<AGGREGATION>>
            # update global weights
            w_glob = FedAvg(w_locals)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            #record time
            end_time = time.time()
            execution_time += end_time - start_time

            acc_train, loss_train_ = test_fun(net_glob, dataset_train, args)
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            total_loss_list[iter] += loss_avg
            print('Round {:3d}, Average training loss {:.3f}'.format(iter, loss_avg))
            print('---\n')
            #end of Epoch


        # testing
        net_glob.eval()

        acc_train, loss_train = test_fun(net_glob, dataset_train, args)
        acc_test, loss_test = test_fun(net_glob, dataset_test, args)

        #accumulation
        total_time += execution_time
        total_acc_train += acc_train
        total_acc_test += acc_test
        total_attack_success_rate += sum(att_acc_list) / len(att_acc_list)

    # experiment setting
    exp_details(args)

    print('Experimental result summary:')
    print(f'args.encrypt_percent:{args.encrypt_percent}')

    print('Execution time: {:.2f}'.format(total_time / duplication))
    print("Training accuracy of the joint model: {:.2f}".format(total_acc_train / duplication))
    print("Testing accuracy of the joint model: {:.2f}".format(total_acc_test / duplication))
    print('Average attack success rate: {:.2f}'.format(total_attack_success_rate / duplication))
    final_total_loss_list = [element / duplication for element in total_loss_list]
    print(f'Average training loss of {args.epochs} iterations:')
    print(', '.join(['{:.16f}'.format(element) for element in final_total_loss_list]))


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(f'args.device:', {args.device})

    sys.stdout = open(f'main_fed_pure.txt', 'w')
    process(args)
    print(f'===========================================\n')
    sys.stdout.close()
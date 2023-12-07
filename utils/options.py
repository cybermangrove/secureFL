import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=20, help="Global rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users in FL: K")
    parser.add_argument('--num_samples', type=int, default=1000,
                        help="number of training samples selected from each local training set")
    parser.add_argument('--alpha', type=float, default=10, help="level of non-iid data distribution: alpha")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=12, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--model', type=str, default='mlp', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='Synthetic', help="name of dataset")
    parser.add_argument('--iid', default=False, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes in the dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--all_clients', default=True, action='store_true', help='aggregation over all clients')

    #HE & DP & Simulation Attack
    parser.add_argument('--encrypt_percent', type=float, default=0.5, help="HE encrypted percentage(%)")
    parser.add_argument('--server_choices', type=int, default=20, help="choice numbers of encrypted index that server giving to clients")
    parser.add_argument('--num_simulations', type=int, default=10, help="number of training samples for simulating attacks")
    parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon for differential privacy. 0.01 or 0.1 provide strong DP protection but inaccurate; 0.7 or 1 provide weak DP protection but accurate")
    args = parser.parse_args()
    return args

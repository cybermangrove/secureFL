import random
from collections import defaultdict

import numpy as np
import torch
import sys

def build_classes_dict(dataset):
    classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if torch.is_tensor(label):
            label = label.numpy()[0]
        else:
            label = label
        if label in classes:
            classes[label].append(ind)
        else:
            classes[label] = [ind]
    return classes


def sample_dirichlet_train_data(dataset, num_participants, num_samples, num_simulations, alpha=0.1):
    data_classes = build_classes_dict(dataset)
    class_size = len(data_classes[0])
    per_participant_list = defaultdict(list)
    per_samples_list = defaultdict(list) # randomly select training samples for evaluating source inference attacks
    per_simulation_list = defaultdict(list) # randomly select training samples for simulating source inference attacks
    no_classes = len(data_classes.keys())
    image_nums = []
    for n in range(no_classes):
        image_num = []
        random.shuffle(data_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(num_participants * [alpha]))
        for user in range(num_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = data_classes[n][:min(len(data_classes[n]), no_imgs)]
            image_num.append(len(sampled_list))
            per_participant_list[user].extend(sampled_list)
            data_classes[n] = data_classes[n][min(len(data_classes[n]), no_imgs):]
        image_nums.append(image_num)

    for i in range(len(per_participant_list)):
        num_samples = min(num_samples, len(per_participant_list[i]))
        num_simulations = min(num_simulations, len(per_participant_list[i]))

    for i in range(len(per_participant_list)):
        sample_index = np.random.choice(len(per_participant_list[i]), num_samples, replace=False)
        per_samples_list[i].extend(np.array(per_participant_list[i])[sample_index])
        simulation_index = np.random.choice(len(per_participant_list[i]), num_simulations, replace=False)
        per_simulation_list[i].extend(np.array(per_participant_list[i])[simulation_index])

    return per_participant_list, per_samples_list, per_simulation_list


def synthetic_iid(dataset, num_users, num_samples, num_simulations):
    num_items = int(len(dataset) / num_users)
    per_participant_list = defaultdict(list)
    all_idxs = [i for i in range(len(dataset))]
    per_samples_list = defaultdict(list)
    dict_simulation_user = defaultdict(list)

    for i in range(num_users):
        per_participant_list[i].extend(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(per_participant_list[i]))

    for i in range(len(per_participant_list)):
        sample_index = np.random.choice(len(per_participant_list[i]), min(len(per_participant_list[i]), num_samples), replace=False)
        per_samples_list[i].extend(np.array(per_participant_list[i])[sample_index])
        simulation_index = np.random.choice(len(per_participant_list[i]), min(len(per_participant_list[i]), num_simulations), replace=False)
        dict_simulation_user[i].extend(np.array(per_participant_list[i])[simulation_index])

    return per_participant_list, per_samples_list, dict_simulation_user


def split_params(w_params_all, encrypted_index=None):
    if encrypted_index is None:
        return w_params_all, []

    need_encrypted_params = []
    non_encrypted_params = []
    SIA_guessed_params = []
    for i in range(len(w_params_all)):
        if i in encrypted_index:
            need_encrypted_params.append(w_params_all[i])
            SIA_guessed_params.append(random.uniform(-2,2))
        else:
            non_encrypted_params.append(w_params_all[i])
    
    return need_encrypted_params, non_encrypted_params, SIA_guessed_params


def generate_full_param(encrypted_index, encrypted_w_param, noised_w_param):
    original_length  = len(encrypted_w_param) + len(noised_w_param)
    obfuscated_list = [None] * original_length

    for i, v in zip(encrypted_index, encrypted_w_param):
        obfuscated_list[i] = v

    noise_iter = iter(noised_w_param)
    for i in range(original_length):
        if obfuscated_list[i] is None:
            obfuscated_list[i] = next(noise_iter)

    return obfuscated_list
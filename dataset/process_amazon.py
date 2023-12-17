import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm

'''
domains = ["fashion", "Beauty", "App", "Arts", "Auto", "Books", "CDs", "Cell", "Clothing",\
           "Music", "Elec", "Gift", "Food", "Home", "Industrial", "Kindle", "Luxury", "Magazine", \
            "Movies", "Instruments", "Office", "Garden", "Pet", "Pantry", "Software", "Sports",\
            "Tools", "Toys", "Video"]
'''
domains = ["Industrial", "Pantry", "Instruments", "Arts", "Office"]


def load_ratings(domains):
    users, items, inters = set(), set(), set()
    item_domain = dict()
    for domain in domains:
        print('Load dataset: ', domain)
        file = os.path.join('./', domain, domain+'.json')
        with open(file, 'r') as fp:
            for line in tqdm(fp, desc='Load ratings'):
                content = json.loads(line)
                try:
                    user = content['reviewerID']
                    item = content['asin']
                    rating = content['overall']
                    time = content['unixReviewTime']
                    users.add(user)
                    items.add(item)
                    item_domain[item] = domain
                    inters.add((user, item, float(rating), int(time)))
                except ValueError:
                    print(line)
    print('Load dataset complete!')
    return users, items, inters, item_domain


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def write_remap_item_index(item2index, item2domain, file):
    print('Write item2domain into file!')
    
    with open(file, 'w')as file:
        for item in item2index:
            item_id = item2index[item]
            item_domain = item2domain[item]
            file.write(str(item)+'\t'+str(item_id)+'\t'+str(item_domain))
            file.write('\n')


def filter_inters(inters, can_items=None,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []

    # filter by meta items
    if can_items:
        print('\nFiltering by meta items: ')
        for unit in inters:
            if unit[1] in can_items:
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))

    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        new_user2count = collections.defaultdict(int)
        new_item2count = collections.defaultdict(int)
        users, n_filtered_users = generate_candidates(
            user2count, user_k_core_threshold)
        items, n_filtered_items = generate_candidates(
            item2count, item_k_core_threshold)
        for unit in inters:
            if unit[0] in users and unit[1] in items:
                new_inters.append(unit)
                new_user2count[unit[0]] += 1
                new_item2count[unit[1]] += 1
        idx += 1
        inters, new_inters = new_inters, []
        user2count, item2count = new_user2count, new_item2count
        print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                % (idx, len(inters), len(user2count), len(item2count)))
        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters


def preprocess_rating(args):
    dataset_full_name = args.dataset

    print('Process rating data: ')
    print(' Dataset: ', dataset_full_name)

    # load ratings
    #rating_file_path = os.path.join(args.input_path, dataset_full_name, dataset_full_name+'.json')
    rating_users, rating_items, rating_inters, item2domain = load_ratings(domains)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print('The number of raw inters: ', len(rating_inters))
    rating_inters = filter_inters(rating_inters, can_items=None,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    # return: list of (user_ID, item_ID, rating, timestamp)
    return rating_inters, item2domain


def get_user_item_from_ratings(ratings):
    users, items = set(), set()
    for line in ratings:
        user, item, rating, time = line
        users.add(user)
        items.add(item)
    return users, items


def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, rating, timestamp = inter
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])
    return user2items, user2index, item2index


def generate_training_data(args, rating_inters, user_ratio=0.8):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)

    # generate train valid test
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters, trajectory_inters, cold_inters = dict(), dict(), dict(), dict(), dict()
    import numpy as np
    idx = np.arange(len(user2index))  # or x.shape[1]
    np.random.shuffle(idx)
    train_idx = idx[:int(len(user2index)*user_ratio)]
    print('total user nums', str(len(user2index)))
    print('train user nums', str(len(train_idx)))
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        # leave one out
        if u_index in train_idx:
            if len(inters) > 20:
                trajectory_inters[u_index] = [str(i_index) for i_index in inters[-10:]]
                train_inters[u_index] = [str(i_index) for i_index in inters[:-12]]
                #print('user inter len', len(trajectory_inters[u_index]))
                #print('train inter len', len(train_inters[u_index]))

                valid_inters[u_index] = [str(inters[-12])]
                test_inters[u_index] = [str(inters[-11])]
                assert len(user2items[u_index]) == len(trajectory_inters[u_index]) + len(train_inters[u_index]) + \
                   len(valid_inters[u_index]) + len(test_inters[u_index])
            else:
                train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
                valid_inters[u_index] = [str(inters[-2])]
                test_inters[u_index] = [str(inters[-1])]
                assert len(user2items[u_index]) == len(train_inters[u_index]) + \
                   len(valid_inters[u_index]) + len(test_inters[u_index])
        else:
            if len(inters) > 50:
                cold_inters[u_index] = [str(i_index) for i_index in inters[-51:]]

    return train_inters, valid_inters, test_inters, user2index, item2index, trajectory_inters, cold_inters



def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index


def write_remap_user_index(unit2index, file):
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')


def convert_to_inter_files(args, rating_inters):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    for u_index in range(len(user2index)):
        inters = user2items[u_index]


def convert_to_atomic_files(args, train_data, valid_data, test_data, trajectory_data, cold_data):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))
    trajectory_uidlist = list(trajectory_data.keys())
    trajectory_uidlist.sort(key=lambda t: int(t))
    cold_uidlist = list(cold_data.keys())
    cold_uidlist.sort(key=lambda t: int(t))
    train_item_set = set()

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]
                train_item_set.add(target_item)
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            target_item = valid_data[uid][0]
            train_item_set.add(target_item)
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            #if target_item in train_item_set:
            #    file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.trajectory.inter'), 'w') as file:
        #file.write('user_id:token\titem_id_list:token_seq\n')
        for uid in trajectory_uidlist:
            item_seq = trajectory_data[uid]
            tgt_seq = item_seq[-10:]
            src_seq = item_seq[:-10]
            in_flag = True
            for item in item_seq:
                if item not in train_item_set:
                    in_flag = False
            if in_flag:
                file.write(f'{" ".join(src_seq)}\t{" ".join(tgt_seq)}\n')

    import csv
    data_list = [['user_id', 'item_id_list', 'item_id']]
    for uid in cold_uidlist:
        item_seq = cold_data[uid]
        in_flag = True
        for item in item_seq:
            if item not in train_item_set:
                in_flag = False
        target_item = item_seq[-1]
        if in_flag:
            data_list.append([str(uid), " ".join(item_seq[:-1]), str(target_item)])
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.cold_test_50.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data_list:
            writer.writerow(row)
    

    data_list = [['user_id', 'item_id_list', 'item_id']]
    for uid in cold_uidlist:
        item_seq = cold_data[uid][-41:]
        in_flag = True
        for item in item_seq:
            if item not in train_item_set:
                in_flag = False
        target_item = item_seq[-1]
        if in_flag:
            data_list.append([str(uid), " ".join(item_seq[:-1]), str(target_item)])
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.cold_test_40.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data_list:
            writer.writerow(row)


    data_list = [['user_id', 'item_id_list', 'item_id']]
    for uid in cold_uidlist:
        item_seq = cold_data[uid][-31:]
        in_flag = True
        for item in item_seq:
            if item not in train_item_set:
                in_flag = False
        target_item = item_seq[-1]
        if in_flag:
            data_list.append([str(uid), " ".join(item_seq[:-1]), str(target_item)])
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.cold_test_30.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data_list:
            writer.writerow(row)


    data_list = [['user_id', 'item_id_list', 'item_id']]
    for uid in cold_uidlist:
        item_seq = cold_data[uid][-21:]
        in_flag = True
        for item in item_seq:
            if item not in train_item_set:
                in_flag = False
        target_item = item_seq[-1]
        if in_flag:
            data_list.append([str(uid), " ".join(item_seq[:-1]), str(target_item)])
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.cold_test_20.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data_list:
            writer.writerow(row)


    data_list = [['user_id', 'item_id_list', 'item_id']]
    for uid in cold_uidlist:
        item_seq = cold_data[uid][-11:]
        in_flag = True
        for item in item_seq:
            if item not in train_item_set:
                in_flag = False
        target_item = item_seq[-1]
        if in_flag:
            data_list.append([str(uid), " ".join(item_seq[:-1]), str(target_item)])
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.cold_test_10.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data_list:
            writer.writerow(row)


    data_list = [['user_id', 'item_id_list', 'item_id']]
    for uid in cold_uidlist:
        item_seq = cold_data[uid][-6:]
        in_flag = True
        for item in item_seq:
            if item not in train_item_set:
                in_flag = False
        target_item = item_seq[-1]
        if in_flag:
            data_list.append([str(uid), " ".join(item_seq[:-1]), str(target_item)])
    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.cold_test_5.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data_list:
            writer.writerow(row)

    
def label_domain(domain_file, test_file, mix_output, diff_output):
    import csv
    with open(domain_file, 'r') as f:
        lines = f.readlines()
    id_domain_dic = {}
    for line in lines:
        line = line.strip().split('\t')
        idx = line[1]
        domain = line[2]
        id_domain_dic[idx] = domain
    with open(test_file, 'r') as t_f:
        lines = t_f.readlines()

    m_data_list = [['user_id', 'item_id_list', 'item_id']]
    d_data_list = [['user_id', 'item_id_list', 'item_id']]
    for line in lines[1:]:
        line = line.strip().split('\t')
        user_id = line[0]
        item_seq = line[1].split(' ')
        item_id = line[2]
        mix_flag = False
        target_domain = id_domain_dic[item_id]
        for item in item_seq:
            domain = id_domain_dic[item]
            if domain == target_domain:
                mix_flag = True
                break
        if mix_flag:
            m_data_list.append([user_id, line[1], item_id])
        else:
            d_data_list.append([user_id, line[1], item_id])
    
    with open(mix_output, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in m_data_list:
            writer.writerow(row)

    with open(diff_output, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in d_data_list:
            writer.writerow(row)
            


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Amazon', help='Amazon all domains')
    parser.add_argument('--user_k', type=int, default=30, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=30, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    #convert_inter()
    print('convert complete')
    # load interactions from raw rating file
    #check_domain(args)
    rating_inters, item2domain = preprocess_rating(args)

    # load item text from raw meta data file
    #item_text_list = preprocess_text(args, rating_inters)
    #convert_to_item_files(item_text_list)

    # split train/valid/test
    train_inters, valid_inters, test_inters, user2index, item2index, trajectory_inters, cold_inters = \
        generate_training_data(args, rating_inters)

    # save interaction sequences into atomic files
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters, trajectory_inters, cold_inters)

    # save useful data
    #write_text_file(item_text_list, os.path.join(args.output_path, args.dataset, f'{args.dataset}.text'))
    write_remap_user_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2index'))
    write_remap_item_index(item2index, item2domain, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2index'))

    
    domain_file = os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2index')
    test_file = os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter')
    mix_output = os.path.join(args.output_path, args.dataset, f'{args.dataset}.mix_test.csv')
    diff_output = os.path.join(args.output_path, args.dataset, f'{args.dataset}.diff_test.csv')
    label_domain(domain_file, test_file, mix_output, diff_output)

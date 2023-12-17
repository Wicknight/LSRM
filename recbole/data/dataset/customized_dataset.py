# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.

Customized datasets named ``[Model Name]Dataset`` can be automatically called.
"""

import numpy as np
import torch

from recbole.data.dataset import SequentialDataset
from recbole.data.interaction import Interaction
from recbole.sampler import SeqSampler
from recbole.utils.enum_type import FeatureType


class LSRMDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
    
    def group_pop(self, cut=4):
        #self._change_feat_format()
        _, _, his_len = self.history_user_matrix()
        #print('his len shape', his_len.shape)
        his_len_array = np.array(his_len)
        total_inters = sum(his_len_array)
        sort_pop = np.argsort(-his_len_array)
        group_len = total_inters // cut
        group_list = []
        idx = -1
        for i in range(cut-1):
            pre_idx = idx
            num = 0
            while(num < group_len):
                idx += 1
                num += his_len_array[sort_pop[idx]]
                
            group_list.append(sort_pop[pre_idx+1:idx+1])
        group_list.append(sort_pop[idx+1:])
        return group_list

    def group_test_data(self, group_target_list, test_dataset):
        group_inter_list = []
        for i in range(len(group_target_list)):
            group_inter_list.append([])
        group_inters = []
        #print('len test dataset', len(test_dataset))
        for data in test_dataset:
            #print('data shape', data.shape)
            target_item = data[self.iid_field]
            if target_item.item() in group_target_list[0]:
                group_inter_list[0].append(data)
            elif target_item.item() in group_target_list[1]:
                group_inter_list[1].append(data)
            elif target_item.item() in group_target_list[2]:
                group_inter_list[2].append(data)
            else:
                group_inter_list[-1].append(data)
        for inter_list in group_inter_list:
            new_inter = stack_interactions(inter_list)
            print('new inter', new_inter)
            group_inters.append(self.copy(new_inter))
        
        return group_inters

    def group_user_data(self, file_path, test_dataset):
        group_inter_list = []
        import pandas as pd
        df = pd.read_csv(file_path)
        user_list = []
        for i, user in enumerate(df['user_id']):
            user = str(user)
            user = self.field2token_id['user_id'][user]
            user_list.append(user)
            #user_list.append(i)
        for data in test_dataset:
            user = data[self.uid_field]
            if user.item() in user_list:
                group_inter_list.append(data)
        print('group data len', len(group_inter_list))
        group_inters = self.copy(stack_interactions(group_inter_list))

        return group_inters

    def read_csv_data(self, file_path):
        import pandas as pd
        df = pd.read_csv(file_path)
        new_df = {}
        seq_list = []
        length_list = []
        for seq in df['item_id_list']:
            seq = seq.split(' ')
            new_seq = []
            for item in seq:
                #print('before convert item', item)
                item = self.field2token_id['item_id'][item]
                #print('after convert item', item)
                new_seq.append(item)
            #print('type item', type(new_seq[0]))
            seq_list.append(new_seq)
            length_list.append(len(new_seq))
        new_df["item_id_list"] = seq_list
        new_df['item_length'] = length_list
        item_list = []
        for item in df['item_id']:
            #print('item type', type(item))
            item = str(item)
            item = self.field2token_id['item_id'][item]
            item_list.append(item)
        new_df['item_id'] = item_list
        user_list = []
        for i, user in enumerate(df['user_id']):
            user = str(user)
            user = self.field2token_id['user_id'][user]
            user_list.append(user)
            #user_list.append(i)
        new_df['user_id'] = user_list
        new_inter = Interaction(new_df)
        #print('new inter', new_inter)
        nxt = self.copy(new_inter)
        
        return nxt
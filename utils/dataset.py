from __future__ import print_function, division
import os
import math
import numpy as np
import pandas as pd

import torch
import json
import time

from torch.utils.data import Dataset
from .encoders.get_conch import CONCHTokenizer
from .encoders.get_qwen import QwenTokenizer
from math import ceil
import random


eps = 1e-5

def is_nan(s):
    try:
        num = float(s)
        return math.isnan(num)
    except ValueError:
        return False


def read_csv_with_encodings(data_csv):
    """尝试多种编码读取CSV文件"""
    encodings = ['utf-8-sig','utf-8', 'latin', 'latin1', 'gbk']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(data_csv, encoding=encoding)
            print(f"Successfully read with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue
    
    raise ValueError(f"Could not read {data_csv} with any encoding")


class BuildDataset(Dataset):
    def __init__(self, args=''):
        self.args = args
        self.stage = self.args.stage
        self.modal = args.modal
        self.phase = 'train'
        self.cohort = args.cohort
        self.endpoint = args.endpoint
        self.dataset_name = args.dataset_name

        with open('data/cancer_type.txt', 'r') as f:
            self.cancer_index = [cancer.strip() for cancer in f.readlines()]
            
        data_csv = f'downstream/{self.cohort}/{self.cohort}.csv'
        #self.data_list = pd.read_csv(data_csv, encoding='latin')
        self.data_list = read_csv_with_encodings(data_csv)
        if self.endpoint in ['os', 'dfs', 'dss', 'dm', 'lr', 'bcr', 'pfs']:
            self.data_list = self.data_list.dropna(subset=[f'{self.endpoint}_status'])
            if f'{args.endpoint}_label' not in self.data_list.columns:
                self.percentile_by_cls(args.endpoint, ncls=5)
        print('Totally {} samples loaded.'.format(len(self.data_list)))
        #external_folds = unique(self.data_list['fold']).remove(0)
        
        if 'split' not in self.data_list.columns:
            num = ceil(len(self.data_list) / 5)
            split = np.tile(np.arange(5), num)[:len(self.data_list)].tolist()
            random.shuffle(split)
            print(split[:10])
            self.data_list['split'] = split
            self.data_list.to_csv(data_csv, index=False)  # Save the updated DataFrame back to the CSV file
        
        self.set_train_test(fold=0)
        self.external_folds = [f for f in np.unique(self.data_list['split']) if f not in [0, 1, 2, 3, 4]]
        self.externals = {}
        for fold_num in self.external_folds:
            self.externals[fold_num] = self.data_list[self.data_list['split'] == fold_num]
            #self.externals.append(self.data_list[self.data_list['split'] == fold_num])
        
        
        if 'text' in args.modal:
            self.report_extractor = CONCHTokenizer()
        
        if 'clin' in args.modal:
            self.clinical_extractor = QwenTokenizer()
            
    def percentile_by_cls(self, endpoint='os', ncls=10):
        self.data_list.dropna(subset=[endpoint + '_status'], inplace=True)
        
        uncensored_df = self.data_list[self.data_list[endpoint + '_status'] > 0]

        tar_column = endpoint + '_time'
        disc_labels, q_bins = pd.qcut(uncensored_df[tar_column], q=ncls, retbins=True, labels=False)
        q_bins[-1] = self.data_list[tar_column].max() + eps
        q_bins[0] = self.data_list[tar_column].min() - eps

        disc_labels, q_bins = pd.cut(self.data_list[tar_column], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        self.data_list[endpoint + '_label'] = disc_labels.values.astype(int)
            
    def set_train_test(self, fold=0):
        self.fold = fold
        self.train_list = self.data_list[(self.data_list['split'] != fold) & (self.data_list['split'].isin([0, 1, 2, 3, 4]))]
        if self.args.per < 1.0:
            train_len = int(len(self.train_list) * self.args.per)
            self.train_list = self.train_list[:train_len]
        self.test_list = self.data_list[self.data_list['split'] == fold]

    def __iter__(self):
        self.test_idx = 0
        return self
    
    def __next__(self):
        cur_idx = self.test_idx
        self.test_idx += 1
        return self.get_sample(cur_idx)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index):
        if self.phase == 'train':
            data_row = self.train_list.iloc[index]
        elif self.phase == 'test':
            data_row = self.test_list.iloc[index]
        else:
            external_idx = int(self.phase)
            data_row = self.externals[external_idx].iloc[index]

        #print(data_row)
        pid = data_row['patient_id']
        ##################### Feature loading #########################
        feat_dict = {}
        if 'path' in self.modal and 'path' in self.train_list.columns and pd.notna(data_row['path']):
            path_files = data_row['path'].split(';')
            if len(path_files) > 3:
                path_files = random.sample(path_files, 3)
            path_feat = []
            try:
                for pfile in path_files:
                    path_feat.append(torch.load(pfile, map_location='cpu'))
                    
                
            except Exception as e:
                print(f'Error loading path files: {e}')
            if len(path_feat) > 0:
                #path_feat = [torch.load(pfile, map_location='cpu') for pfile in path_files]
                pfeat = torch.concat(path_feat, dim=0).float()
                if len(pfeat) > 20000:
                    pfeat = pfeat[:20000]
                feat_dict['path'] = pfeat
        
        
        if 'ihc' in self.modal and 'ihc' in self.train_list.columns and pd.notna(data_row['ihc']):
            ihc_files = data_row['ihc'].split(';')
            try:
                ihc_feat = [torch.load(ifile, map_location='cpu') for ifile in ihc_files]
            except Exception as e:
                print(f'Error loading ihc files: {e}')
            ifeat = torch.concat(ihc_feat, dim=0).float()
            if len(ifeat) > 20000:
                ifeat = ifeat[:20000]
            feat_dict['ihc'] = ifeat
        
        
        if 'gene' in self.modal and 'gene' in self.train_list.columns and pd.notna(data_row['gene']):
            gene_file = data_row['gene']
            if gene_file.endswith('.pt'):
                if os.path.exists(gene_file):
                    feat_dict['gene'] = torch.load(gene_file, map_location='cpu').float()
                #feat_dict['gene'] = torch.load(gene_file, map_location='cpu').float()
            elif gene_file.endswith('.npy'):
                feat_dict['gene'] = torch.tensor(np.load(gene_file)).float()
            else:
                print(f'Unknown gene feature files: {gene_file}')
        
        
        if 'text' in self.modal and 'text' in self.train_list.columns:
            if 'text_pt' in self.train_list.columns and pd.notna(data_row['text_pt']):
                feat_dict['text'] = torch.load(data_row['text_pt'], map_location='cpu').float()
            elif pd.notna(data_row['text']):
                feat_dict['text'] = self.report_extractor.get_token([data_row['text']])
        
        if 'clin' in self.modal and 'clin' in self.train_list.columns:
            if 'clin_pt' in self.train_list.columns and pd.notna(data_row['clin_pt']):
                feat_dict['clin'] = torch.load(data_row['clin_pt'], map_location='cpu').float()
            elif pd.notna(data_row['clin']):
                clin_feat = self.clinical_extractor.get_token(data_row['clin'])
                feat_dict['clin_input_ids'] = clin_feat['input_ids']
                feat_dict['clin_attention_mask'] = clin_feat['attention_mask']
        
        
        #if len(feat_dict.keys()) < 2:
        #    print(data_row['patient_id'], data_row['cohort'], feat_dict.keys())
            
        feat_dict['cid'] = self.cancer_index.index(data_row['site']) #torch.tensor(self.args.cancer_id)
        
        ##################### Label formating #########################
        label_dict ={}
        if self.args.task == 'cls':
            label_dict['label'] = torch.tensor(data_row[self.endpoint]).view(-1)
        elif self.args.task == 'bcls':
            label_dict['label'] = torch.tensor(data_row[self.endpoint]).view(-1)
        else:
            label_dict['surv_label'] = torch.tensor(int(data_row[f'{self.endpoint}_label'])).view(-1)
            label_dict['surv_time'] = torch.tensor(float(data_row[f'{self.endpoint}_time'])).view(-1)
            label_dict['surv_status'] = torch.tensor(1 - float(data_row[f'{self.endpoint}_status'])).view(-1)
        
        
        return feat_dict, label_dict, pid
    
    def __len__(self):
        if self.phase == 'train':
            return len(self.train_list)
        elif self.phase == 'test':
            return len(self.test_list)
        else:
            return len(self.externals[int(self.phase)])

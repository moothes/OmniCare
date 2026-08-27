from __future__ import print_function, division
import os
import math
import random
import json
import numpy as np
import pandas as pd

import torch
import json

#from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer
from torch.utils.data import Dataset
from .encoders.get_conch import CONCHTokenizer
from .encoders.get_qwen import QwenTokenizer

#import torch.multiprocessing as mp
#mp.set_start_method('spawn', force=True)

eps = 1e-5

def is_nan(s):
    try:
        num = float(s)
        return math.isnan(num)
    except ValueError:
        return False
        
def get_pretrain_data():
    #fmt = "data/data_list/{}_datalist_with_embedding.csv"
    fmt = "data/data_list/final/{}_datalist.csv"
    #data_list = ["ihc", "tcga", 'prep']
    data_list = ["tcga", ]
    dfs = []
    len_df = {}
    for data_file in data_list:
        data_path = fmt.format(data_file)
        df = pd.read_csv(data_path, encoding='utf-8-sig', dtype=str, index_col=False)
        dfs.append(df)
        len_df[data_file] = len(df)
    print(f"Data lengths: {len_df}")
    return pd.concat(dfs, ignore_index=True)
       
class PretrainDataset(Dataset):
    def __init__(self, args=''):
        self.args = args
        self.stage = self.args.stage
        self.modal = args.modal
        self.phase = 'train'
        
        with open('data/cancer_type.txt', 'r') as f:
            self.cancer_index = [cancer.strip() for cancer in f.readlines()]

        self.data_list = get_pretrain_data()
        
        
        if 'text' in args.modal:
            # Load CONCH for extracting embedding from pathology report
            self.report_extractor = CONCHTokenizer()
        
        if 'clin' in args.modal:
            # Load QWEN3 for extracting embedding from clinical record
            self.clinical_extractor = QwenTokenizer()
        
        
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
        data_row = self.data_list.iloc[index]
        pid = f"{data_row['cohort']}/{data_row['patient_id']}"

        ##################### Feature loading #########################
        feat_dict = {}
        if 'path' in self.modal and pd.notna(data_row['path']):
            path_files = data_row['path'].split(';')
            path_feat = []
            for pfile in path_files:
                if not os.path.exists(pfile):
                    print(f"File not found: {pfile}")
                    continue
                path_feat.append(torch.load(pfile, weights_only=False, map_location='cpu'))
            path_feat = torch.concat(path_feat, dim=0).float()
            
            feat_dict['path'] = path_feat
            #else:
                
            #feat_dict['path'] = torch.zeros([5, 2560])
            '''
            if 'wsi_paths' in data_row.keys():
                if not pd.isna(data_row['wsi_paths']):
                    wsi_files = data_row['wsi_paths'].split(';')
                    path_feat = []
                    for wfile in wsi_files:
                        if not os.path.exists(wfile):
                            print(f"File not found: {wfile}")
                            continue
                        path_feat.append(torch.load(wfile, map_location='cpu'))
                    path_feat = torch.concat(path_feat, dim=0).float()
                    feat_dict['path'] = path_feat
                    
                    
                    feat_dict['path'] = [torch.zeros([5, 2560])]
            '''
        
        if 'ihc' in self.modal and pd.notna(data_row['ihc']):
            ihc_files = data_row['ihc'].split(';')
            ihc_feat = []
            for ifile in ihc_files:
                ihc_feat.append(torch.load(ifile, weights_only=False, map_location='cpu'))
            ihc_feat = torch.concat(ihc_feat, dim=0).float()
            
            feat_dict['ihc'] = ihc_feat
        
        if 'gene' in self.modal and pd.notna(data_row['gene_pt']):
            gene_file = data_row['gene_pt']
            if gene_file.endswith('.pt'):
                gene_feat = torch.load(gene_file, weights_only=False, map_location='cpu').float()
                feat_dict['gene'] = gene_feat
            if gene_file.endswith('.npy'):
                gene_feat = torch.tensor(np.load(gene_file)).float()
                feat_dict['gene'] = gene_feat
        
        if 'text' in self.modal and pd.notna(data_row['text']):
            text_feat = self.report_extractor.get_token([data_row['text']])
            feat_dict['text'] = text_feat
        
        if 'clin' in self.modal and pd.notna(data_row['clin']):
            clin_feat = self.clinical_extractor.get_token(data_row['clin'])
            feat_dict['clin_input_ids'] = clin_feat['input_ids']
            feat_dict['clin_attention_mask'] = clin_feat['attention_mask']
        
        if len(feat_dict.keys()) < 2:
            print(data_row['patient_id'], data_row['cohort'], feat_dict.keys())

        # cancer id specific to dataset, please find it in config/dataset
        feat_dict['cid'] = self.cancer_index.index(data_row['site']) #torch.tensor(self.args.cancer_id)
        
        ##################### Label formating #########################
        label_dict ={}
        if pd.notna(data_row['OS_label']):
            label_dict['surv_label'] = torch.tensor(int(data_row['OS_label']))
            label_dict['surv_time'] = torch.tensor(float(data_row['Event']))
            label_dict['surv_status'] = torch.tensor(1 - float(data_row['Status']))
        # slide_level IHC label
        if 'ihc_pt' in self.data_list.columns and pd.notna(data_row['ihc_pt']):
            label_dict['ihc_pt'] = torch.load(data_row['ihc_pt'], weights_only=False, map_location='cpu').float()
        if 'text_pt' in self.data_list.columns and pd.notna(data_row['text_pt']):
            label_dict['text_pt'] = torch.load(data_row['text_pt'], weights_only=False, map_location='cpu').float()
        if 'clin_pt' in self.data_list.columns and pd.notna(data_row['clin_pt']):
            label_dict['clin_pt'] = torch.load(data_row['clin_pt'], weights_only=False, map_location='cpu').float()

        return feat_dict, label_dict, pid
    
    def __len__(self):
        return len(self.data_list)

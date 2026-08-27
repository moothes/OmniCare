import os
import sys
import csv
import time
import random
import torch
import numpy as np
import pandas as pd
import importlib

from shutil import copyfile

from utils.options import parse_args
from utils.pretrain_engine import PretrainEngine
from utils.pretrain_dataset import PretrainDataset
from utils.utils import *

def set_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def main(args):    
    dataset = PretrainDataset(args)
    args.results_dir = get_save_path(args)
        
    dfs = []    
    for fold in args.folds:
        args.current_fold = fold
        set_seed(args.seed)
        #dataset.set_train_test(fold)
        engine = PretrainEngine(args)
        
        result = engine.learning(dataset)
        res_dict = pd.DataFrame([result])
        dfs.append(res_dict)
        print('Overall: {}.'.format(result))
    
    summary_df = pd.concat(dfs, ignore_index=True)
    summary_df.to_csv(os.path.join(args.results_dir, "results_summary.csv"), index=False)
    return summary_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Need model name!')
        exit()
        
    args = parse_args()
    print(args)
    results = main(args)
    print(results)
    print("Finished!")

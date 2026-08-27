import os
import sys
import time
import random
import torch
import numpy as np
import pandas as pd
import importlib

from shutil import copyfile

from utils.options import parse_args
from utils.engine import Engine
from utils.dataset import BuildDataset

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
    dataset = BuildDataset(args)
    print(args)

    results_dir = f"./results/{args.model}/finetune/{args.cohort}"
    if args.per < 1:
        results_dir = results_dir + f'_{args.per}'
    #results_dir = f"./results/{args.model}/finetune/{args.cohort}_{args.per}_{time.strftime('%Y-%m-%d]-[%H-%M-%S')}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    args.results_dir = results_dir
    csv_path = os.path.join(results_dir, f"{args.model}_{args.cohort}_{args.per}_results.csv")
    
    summary_df = []
    all_folds = list(map(int, args.folds))
    
    weights_files = [weights for weights in os.listdir(results_dir) if weights.endswith('.pth.tar')]
    for weights_file in weights_files:
        fold = int(weights_file.split('_')[1])  # Assuming the fold number is the second part of the filename
        #assert fold in all_folds, f"Fold {fold} in weights file {weights_file} is not in the specified folds."
        if fold in all_folds:
            all_folds.remove(fold)  # Remove the fold from the list of folds to process
            
            args.current_fold = fold
            set_seed(args.seed)
            dataset.set_train_test(fold)
            
            weight_path = os.path.join(results_dir, weights_file)
            engine = Engine(args, weights=weight_path)
            result = engine.deploy(dataset)
            result['fold'] = fold
            
            res_dict = pd.DataFrame([result])
            summary_df.append(res_dict)
            
            print(f'Fold {fold} results: {result}.')
        
    for fold in all_folds:
        args.current_fold = fold
        set_seed(args.seed)
        dataset.set_train_test(fold)
        
        weight_path = None
        if args.model == "OmniCare":
            weight_path = 'downstream/OmniCare.pth.tar'
        
        engine = Engine(args, weights=weight_path)
        result = engine.learning(dataset)
        result['fold'] = fold
        
        res_dict = pd.DataFrame([result])
        summary_df.append(res_dict)
        
        print(f'Fold {fold} results: {result}.')
    
    sdf = pd.concat(summary_df)
    sdf.to_csv(csv_path, index=False)
    
    return csv_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Need model name!')
        exit()
        
    args = parse_args()
    main(args)
    print("Finished!")

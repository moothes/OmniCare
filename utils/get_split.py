import os
import pandas as pd
import numpy as np
import math
import random

'''
base_path = '/home/fengtao/code/TCGA-All/'
writer = pd.ExcelWriter('../../data/tcga-all—new.xlsx', engine='xlsxwriter')

subsets = os.listdir(base_path)#[:1]
for subset in subsets:
    anno = pd.read_csv(base_path + subset)
    #print(anno.shape)
    anno = anno.dropna(subset=['Status', 'WSI', 'RNA'])
    #print(anno.shape)

    split = np.tile(range(5), math.ceil(anno.shape[0] / 5.))[:anno.shape[0]]
    random.shuffle(split)
    anno['split'] = split
    print(anno.shape)
    anno.to_excel(writer, sheet_name='TCGA-' + subset.split('.')[0], index=False)

writer._save()
'''


base_path = '/home/fengtao/code/TCGA-All/'
save_rile = '../../data/tcga-all-new.csv'

df = None
subsets = os.listdir(base_path)
for subset in subsets:
    anno = pd.read_csv(base_path + subset)
    anno = anno.dropna(subset=['Status', 'WSI', 'RNA'])

    split = np.tile(range(5), math.ceil(anno.shape[0] / 5.))[:anno.shape[0]]
    random.shuffle(split)
    anno['split'] = split
    print(anno.shape)
    if df is None:
        df = anno
    else:
        df = pd.concat([df, anno])

df.to_csv(save_rile)
print(df.shape)

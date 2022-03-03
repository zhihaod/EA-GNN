import numpy as np
import sys
from utils import *
import matplotlib.pyplot as plt
import networkx as nx
from ncgnn.utils import *
from train1 import train,train_model,train_model_gat,train_model_multi
from ncgnn.model_builder import create_model
import torch
from ncgnn.config import cfg
import argparse
import time




parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='ncgnn')
#parser.add_argument('--subsample', type=str, default='plain')
args = parser.parse_args()
model = args.model
#feature_select = args.subsample

cfg_names = ['cora','citeseer','pubmed','academic_cs','academic_py','amazon_computer','amazon_photo']
task_path = '/home/zhihao/Document/gnn_fd/graphSage/split/multihead/NC-GNN/'
for cfg_name in cfg_names:
    print(f'datasaet: {cfg_name},  model: {model}.')
    cfg_file = task_path + f'cfg/shuffling/' + cfg_name + '.yaml'
    cfg.merge_from_file(cfg_file)
    path = cfg.dataset.path
    path_result = cfg.task_path + 'results/'  
    ratio_train_val_test = cfg.dataset.split
    batch_size = cfg.train.batch_size
    lr = cfg.optim.lr
    cuda= cfg.model.cuda
    beta = cfg.model.beta
    interval = cfg.model.interval
    r = cfg.model.r
    patience = cfg.train.patience
    gcn=cfg.model.gcn
    epochs = cfg.optim.epochs
    repeat = cfg.train.repeat
    hidden=cfg.model.hidden
    dataset = cfg.dataset.name
    num_sample =cfg.model.num_sample
    #compress_ratios = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    shuffle_ratios = [0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.0]
   
    if model == 'sage':
        beta = 0.0
        interval = 0.0
        r = 1    
    
    results = []
    for ratio in shuffle_ratios:
        
        features,labels,full_adjs_dict = load_dataset(dataset,path)    
        
        full_adjs_shuffled = shuffle_edges(full_adjs_dict,ratio,10)
       # feat_data_compress = feature_subsampling[feature_select](features,ratio,labels,1)
        
        result_ = []
        for seed in range(repeat):
            time_start = time.time()
            train_idx,val_idx,test_idx = create_data_splits(labels,seed,ratio_train_val_test)
            result = train_model(features,labels,num_sample,hidden,cuda,gcn,beta,lr,train_idx,val_idx,test_idx,full_adjs_shuffled,batch_size,epochs,dataset,patience,r,interval)
                #print(result)
            result_.append(result[2])
            time_end = time.time()
            print(f'time of one repeat:{time_end-time_start}')
        results.append(result_)
    save_list(path_result+f'{dataset}/'+f'{model}_{num_sample}_edge_shuffle.pkl',results)  
    

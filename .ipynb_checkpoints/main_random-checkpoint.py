import numpy as np
import random
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





task_path = '/home/zhihao/Document/gnn_fd/graphSage/split/multihead/NCGNN_reorderedge/'
cfg_names = ['cora','citeseer','pubmed','academic_cs','academic_py','amazon_computer','amazon_photo']
for cfg_name in cfg_names:
    print(cfg_name)
    cfg_file = task_path + 'cfg/ncgnn/random/' + cfg_name + '.yaml'
    cfg.merge_from_file(cfg_file)
    path = cfg.dataset.path
    path_result = cfg.path_result
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
    model_name = cfg.model.name
    num_sample =cfg.model.num_sample
    feature_select = cfg.dataset.feature_select
    compress_ratios = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    
    results = []
    for ratio in compress_ratios:
    
        if dataset in ['cora','citeseer','pubmed','academic_cs','academic_py']:
            
            features,labels,full_adjs_dict =load_data(dataset,path=path)
            #feat_data_compress = features[:,:int(features.shape[1]*ratio)]     
            random.seed(10)
            inds = random.sample(range(features.shape[1]),int(features.shape[1]*ratio))
            feat_data_compress = features[:,inds]
            
            result_ = []
            for seed in range(repeat):
                time_start = time.time()
                train_idx,val_idx,test_idx = create_data_splits(labels,seed,ratio_train_val_test)
                result = train_model(feat_data_compress,labels,num_sample,hidden,cuda,gcn,beta,lr,train_idx,val_idx,test_idx,full_adjs_dict,batch_size,epochs,dataset,patience,r,interval)
                #print(result)
                result_.append(result[2])
                time_end = time.time()
                print(f'time of one repeat:{time_end-time_start}')
            results.append(result_)
            #save_list(path_result+f'{dataset}/'+f'{model_name}_{num_sample}.pkl',results)
            
        elif dataset in ['amazon_computer','amazon_photo']:         
            features,labels,adj_mat =load_data_npz(dataset,path=path)       
            #feat_data_compress = features[:,:int(features.shape[1]*ratio)]
            random.seed(10)
            inds = random.sample(range(features.shape[1]),int(features.shape[1]*ratio))
            feat_data_compress = features[:,inds]
            
            
            result_ = []
            for seed in range(repeat):   
                time_start = time.time()
                train_idx,val_idx,test_idx,full_adjs_dict = create_data_splits_mat(labels,adj_mat,seed,ratio_train_val_test)
                result = train_model(feat_data_compress,labels,num_sample,hidden,cuda,gcn,beta,lr,train_idx,val_idx,test_idx,full_adjs_dict,batch_size,epochs,dataset,patience,r,interval)
                #print(result)
                result_.append(result[2])
                time_end = time.time()
                print(f'time of one repeat:{time_end-time_start}')
            results.append(result_)
           # save_list(path_result+f'{dataset}/'+f'{model_name}_{num_sample}.pkl',results)   
        else:
            print('Dataset does not exist.')
    save_list(path_result+f'{dataset}/'+f'{model_name}_{num_sample}.pkl',results)  
    
    
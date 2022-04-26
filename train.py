import glob
import os
import copy
import time
import torch
from torch.autograd import Variable
import random
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
from ncgnn.encoders import Encoder,Encoder_multi
from ncgnn.model import SupervisedGraphSage,SupervisedGraphSage_multi
from ncgnn.model_builder import create_model,create_model_multi
import torch.nn.functional as F
import time

import networkx as nx

def train_model(features,labels,num_sample,hidden,cuda,gcn,beta,lr,train_idx,val_idx,test_idx,full_adjs_dict,batch_size,epochs,dataset,patience,r,interval):
    model = create_model(features,labels,len(np.unique(labels)),num_sample,hidden,cuda,gcn,beta)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    time_start = time.time()
    full_adjs_dict = create_weighted_adj_dict(full_adjs_dict)
    time_end = time.time()
    print(f'time of construct data:{time_end-time_start}')
    result = train(model,optimizer,train_idx,val_idx,test_idx,full_adjs_dict,features,labels,batch_size,lr,len(np.unique(labels)),epochs,dataset,cuda,gcn,patience)
    print(f'beta:{beta},accuracy:{result[2]}\n')
    temp_beta = beta
    for i in range(1,r):
        beta = beta + interval
        time_start = time.time()
        model = create_model(features,labels,len(np.unique(labels)),num_sample,hidden,cuda,gcn,beta)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        temp = train(model,optimizer,train_idx,val_idx,test_idx,full_adjs_dict,features,labels,batch_size,lr,len(np.unique(labels)),epochs,dataset,cuda,gcn,patience)
        time_end = time.time()
        print(f'time of one train:{time_end-time_start}')
        if result[2] < temp[2]:
            result = temp   
            temp_beta = beta
        print(f'beta:{beta},accuracy:{temp[2]}\n')
    result[2] = (temp_beta,result[2])
    return result

def train_model_multi(features,labels,num_sample,hidden,cuda,gcn,beta,lr,train_idx,val_idx,test_idx,full_adjs_dict,batch_size,epochs,dataset,patience,r,interval):
    model = create_model_multi(features,labels,len(np.unique(labels)),num_sample,hidden,cuda,gcn,beta)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)           
    full_adjs_dict = create_weighted_adj_dict(full_adjs_dict)
    result = train(model,optimizer,train_idx,val_idx,test_idx,full_adjs_dict,features,labels,batch_size,lr,len(np.unique(labels)),epochs,dataset,cuda,gcn,patience)
    print(f'beta:{beta},accuracy:{result[2]}')
    return result




def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def train(model,optimizer,train_idx,val_idx,test_idx,full_adjs_dict,features,labels,batch_size,lr,num_class,epochs,dataset,cuda,gcn,patience):
    
    loss_values = []
    loss_train = []
    bad_counter = 0
    best = epochs + 1
    best_epoch = 0  
    temp_path = '/home/zhihao/temp/intermediate/' + f'{dataset}/'
    #full_adjs_dict = create_weighted_adj_dict(full_adjs_dict)
    time_start = time.time()
    for epoch in range(epochs):  
        train_losses,val_losses = train_epoch(model,optimizer,train_idx,val_idx,test_idx,full_adjs_dict,features,labels,batch_size,lr,cuda)     
        print(f'epoch: {epoch}, val_loss: {np.average(val_losses)}')  
        loss_train.append(np.average(train_losses))
        loss_values.append(np.average(val_losses))  

        torch.save(model.state_dict(),temp_path+'{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch 
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == patience:
            break
        files = glob.glob(temp_path+'*.pkl')
        for file in files:
            epoch_nb = int(file[len(temp_path):].split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)
    print(f'Final epoch: {epoch}')
    result = test(model,labels,temp_path,best_epoch,test_idx,batch_size,num_class,full_adjs_dict)
    return [loss_train,loss_values,result]



def create_weighted_adj_dict(full_adjs):
    G_ = nx.Graph(full_adjs)
    new_adjs = {}
    for node in list(G_.nodes()):
        G_ego = nx.ego_graph(G_,node)
        ego_nodes = list(G_ego.nodes)
        ego_nodes.remove(node)
        new_adjs[node] = set([(n,G_ego.degree(n)) for n in ego_nodes]) 
    return new_adjs



    
def train_epoch(model,optimizer,train_idx,val_idx,test_idx,full_adjs_dict,features,labels,batch_size,lr,cuda):
    random.shuffle(train_idx)
    num_batches = int(len(train_idx) / batch_size) + 1
    train_losses = []
    start_time = time.time()
    for batch in range(num_batches):   
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, len(train_idx))
        batch_nodes = train_idx[i_start:i_end]
        optimizer.zero_grad()
        if cuda:
            loss = model.loss(batch_nodes,full_adjs_dict,Variable(torch.cuda.LongTensor(labels[np.array(batch_nodes)])))
        else:
            loss = model.loss(batch_nodes,full_adjs_dict,Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        train_losses.append(loss.data.item())
        loss.backward()
        optimizer.step()
        

        num_batches_val = int(len(val_idx) / batch_size) 
        val_losses = []
        for batch in range(num_batches_val):
            i_start = batch * batch_size
            i_end = min((batch + 1) * batch_size, len(val_idx))
            batch_nodes = val_idx[i_start:i_end]
            if cuda:
                loss = model.loss(batch_nodes,full_adjs_dict,Variable(torch.cuda.LongTensor(labels[np.array(batch_nodes)])))
            else:
                loss = model.loss(batch_nodes,full_adjs_dict,Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            
            val_losses.append(loss.data.item())
    return train_losses,val_losses    
    

def test(model,labels,temp_path,best_epoch,test_idx,batch_size,num_class,full_adjs_dict):
    files = glob.glob(temp_path+'*.pkl')
    for file in files:
        epoch_nb = int(file[len(temp_path):].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    model.load_state_dict(torch.load(temp_path+'{}.pkl'.format(best_epoch)))        
    num_batches_test = int(len(test_idx) / batch_size) 
    test_outputs = []
    test_labels = []
    for batch in range(num_batches_test):
        i_start = batch * batch_size
        i_end = (batch + 1) * batch_size
        batch_nodes = test_idx[i_start:i_end]
        test_output = model.forward(batch_nodes,full_adjs_dict) 
        test_outputs.extend(list(test_output.data.cpu().numpy().argmax(axis=1)))
        test_labels.extend(list(labels[batch_nodes]))
    return accuracy_score(test_labels,test_outputs)
        
 


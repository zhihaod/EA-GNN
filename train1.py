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
from gat.models import SpGAT
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
    #temp_beta = beta
    #for i in range(1,r):
    #    beta = beta + interval
    #    model = create_model(features,labels,len(np.unique(labels)),num_sample,hidden,cuda,gcn,beta)
    #    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    #    temp = train(model,optimizer,train_idx,val_idx,test_idx,full_adjs_dict,features,labels,batch_size,lr,len(np.unique(labels)),epochs,dataset,cuda,gcn,patience)
    #    if result[2] < temp[2]:
    #        result = temp   
    #        temp_beta = beta
    #    print(f'beta:{beta},accuracy:{result[2]}')
    #result[2] = (temp_beta,result[2])
    return result




def train_model_gat(features,labels,num_sample,hidden,cuda,gcn,beta,lr,train_idx,val_idx,test_idx,full_adjs_dict,batch_size,epochs,dataset,patience,r,interval):
    model = SpGAT(nfeat=features.shape[1], 
                nhid=hidden, 
                nclass=len(np.unique(labels)), 
                nheads=8, 
                dropout=0,
                alpha=0.2)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)    
    features, adj, labels = Variable(features), Variable(full_adjs_dict), Variable(labels)
    if cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        
        train_idx = torch.LongTensor(train_idx).cuda()
        val_idx = torch.LongTensor(val_idx).cuda()
        test_idx = torch.LongTensor(test_idx).cuda()

    
    result = train_gat(model,optimizer,train_idx,val_idx,test_idx,adj,features,labels,batch_size,lr,int(labels.max()) + 1,epochs,dataset,cuda,gcn,patience)
    print(f'beta:{beta},accuracy:{result[2]}')
    temp_beta = beta
    result[2] = (temp_beta,result[2])
    return result




def train_gat(model,optimizer,train_idx,val_idx,test_idx,adj,features,labels,batch_size,lr,num_class,epochs,dataset,cuda,gcn,patience):
    
    loss_values = []
    bad_counter = 0
    best = epochs + 1
    best_epoch = 0  
    temp_path = './intermediate/' + f'{dataset}/'
    #full_adjs_dict = create_weighted_adj_dict(full_adjs_dict)
    
    for epoch in range(epochs):
        
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)

        loss_train = F.nll_loss(output[train_idx], labels[train_idx])
        
        acc_train = accuracy(output[train_idx], labels[train_idx])
        loss_train.backward()
        optimizer.step()
        loss_val = F.nll_loss(output[val_idx], labels[val_idx])
        acc_val = accuracy(output[val_idx], labels[val_idx])        
        
        
        
        loss_values.append(loss_val.data.item())     
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
                
                
    files = glob.glob(temp_path+'*.pkl')
    for file in files:
        epoch_nb = int(file[len(temp_path):].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    model.load_state_dict(torch.load(temp_path+'{}.pkl'.format(best_epoch)))
    
    
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    return [loss_train,loss_values,acc_test.item()]

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
    temp_path = './intermediate/' + f'{dataset}/'
    #full_adjs_dict = create_weighted_adj_dict(full_adjs_dict)
    time_start = time.time()
    for epoch in range(epochs):  
        train_losses,val_losses = train_epoch(model,optimizer,train_idx,val_idx,test_idx,full_adjs_dict,features,labels,batch_size,lr,cuda)     
        print(f'epoch: {epoch}, val_loss: {np.average(val_losses)}')  
        loss_train.append(np.average(train_losses))
        loss_values.append(np.average(val_losses))  
        
        #best,bad_counter,best_epoch = save_model(model,epoch,temp_path,loss_values,best,bad_counter,best_epoch,patience)

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

#def create_weighted_adj_dict(full_adjs):
#    new_adjs = {}
#    print(len(full_adjs))
#    for key in full_adjs:
#        neighbs_set = full_adjs[key] | set([key])
#        n_list = []
#        for node in neighbs_set:
#            n_list.extend(full_adjs[node] & neighbs_set)
#        new_adjs[key] = set()
#        
#        neighbs_set.remove(key)
#        for node in neighbs_set:
#            new_adjs[key].add((node,n_list.count(node)))
#    print(len(full_adjs))
#    return new_adjs

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
        #test_output = torch.zeros(batch_size,num_class)
        #if cuda:
        #    test_output = test_output.cuda()
        #for i in range(3):
        #test_output = test_output + graphsage.forward(batch_nodes,adj_lists) 
        test_output = model.forward(batch_nodes,full_adjs_dict) 
        test_outputs.extend(list(test_output.data.cpu().numpy().argmax(axis=1)))
        test_labels.extend(list(labels[batch_nodes]))
    return accuracy_score(test_labels,test_outputs)
        
#def save_model(model,epoch,temp_path,loss_values,best,bad_counter,best_epoch,patience):
#    torch.save(model.state_dict(),temp_path+'{}.pkl'.format(epoch))
#    if loss_values[-1] < best:
#        best = loss_values[-1]
#        best_epoch = epoch 
#        bad_counter = 0
#    else:
#        bad_counter += 1
#    if bad_counter == patience:
#        break
#    files = glob.glob(temp_path+'*.pkl')
#    for file in files:
#        epoch_nb = int(file[len(temp_path):].split('.')[0])
#        if epoch_nb < best_epoch:
#            os.remove(file)
#    return best,bad_counter,best_epoch
    
    

            

    
    
    
    
    
    
def train_1(cuda,gcn,feat_data, labels, adj_lists,adj_lists_new,valid_adjs,train,valid,test,num_sample_1,num_sample_2,beta_1,epochs,patience,num_class,temp_path,hidden=128,lr=0.0001,batch_size=128):
    np.random.seed(1)
    random.seed(1)

    c_train = copy.deepcopy(train)
    c_test = copy.deepcopy(test)
    c_valid = copy.deepcopy(valid)
    
    num_nodes = labels.shape[0]
    features = torch.nn.Embedding(num_nodes, feat_data.shape[1])
    features.weight = torch.nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    if cuda:
        features.cuda()
    agg1 = WeightedAggregator_new(features, cuda=cuda,gcn=gcn,beta=beta_1,subgraph=True,is_first=True)
    enc1 = Encoder(features, feat_data.shape[1], hidden, aggregator=agg1,num_sample=num_sample_1, gcn=gcn, cuda=cuda)
    agg2 = WeightedAggregator_new(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(), cuda=cuda,gcn=True,beta=beta_1,subgraph=True,is_first=False)
    enc2 = Encoder(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(), enc1.embed_dim, hidden, aggregator=agg2,num_sample=num_sample_2,
            base_model=enc1, gcn=True, cuda=cuda)
    graphsage = SupervisedGraphSage(num_class, enc1)    
    if cuda:
        graphsage.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=lr)
    
    t_total = time.time()
    loss_values = []
    loss_train = []
    bad_counter = 0
    best = epochs + 1
    best_epoch = 0   
    
    for epoch in range(epochs):
        random.shuffle(c_train)
        num_batches = int(len(c_train) / batch_size) + 1
        train_losses = []
        start_time = time.time()
        for batch in range(num_batches):   
            i_start = batch * batch_size
            i_end = min((batch + 1) * batch_size, len(c_train))
            batch_nodes = c_train[i_start:i_end]
            optimizer.zero_grad()
            if cuda:
                loss = graphsage.loss(batch_nodes,adj_lists_new,Variable(torch.cuda.LongTensor(labels[np.array(batch_nodes)])))
            else:
                loss = graphsage.loss(batch_nodes,adj_lists_new,Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            train_losses.append(loss.data.item())
            loss.backward()
            optimizer.step()

        
        num_batches_valid = int(len(c_valid) / batch_size) 
        valid_losses = []
        for batch in range(num_batches_valid):
            i_start = batch * batch_size
            i_end = min((batch + 1) * batch_size, len(c_valid))
            batch_nodes = c_valid[i_start:i_end]
            if cuda:
                loss = graphsage.loss(batch_nodes,valid_adjs,Variable(torch.cuda.LongTensor(labels[np.array(batch_nodes)])))
            else:
                loss = graphsage.loss(batch_nodes,valid_adjs,Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            
            valid_losses.append(loss.data.item())

        loss_train.append(np.average(train_losses))
        loss_values.append(np.average(valid_losses))
        torch.save(graphsage.state_dict(),temp_path+'{}.pkl'.format(epoch))
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
                
                
                
    files = glob.glob(temp_path+'*.pkl')
    for file in files:
        epoch_nb = int(file[len(temp_path):].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
   # print("Optimization Finished!")
    #print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    graphsage.load_state_dict(torch.load(temp_path+'{}.pkl'.format(best_epoch)))        
    #print(best_epoch)
    
    num_batches_test = int(len(c_test) / batch_size) 
    test_outputs = []
    test_labels = []
    for batch in range(num_batches_test):
        i_start = batch * batch_size
        i_end = (batch + 1) * batch_size
        batch_nodes = c_test[i_start:i_end]
        test_output = torch.zeros(batch_size,num_class)
        if cuda:
            test_output = test_output.cuda()
        for i in range(3):
            test_output = test_output + graphsage.forward(batch_nodes,adj_lists) 
        test_outputs.extend(list(test_output.data.cpu().numpy().argmax(axis=1)))
        test_labels.extend(list(labels[batch_nodes]))
    return [loss_train,loss_values,accuracy_score(test_labels,test_outputs)]



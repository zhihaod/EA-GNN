import glob
import os
import copy
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
from sklearn.metrics import f1_score,accuracy_score

from aggregators import MeanAggregator2, WeightedAggregator,WeightedAggregator_new
from encoders import Encoder
from model import SupervisedGraphSage

def train_1(cuda,gcn,feat_data, labels, adj_lists,adj_lists_new,valid_adjs,train,valid,test,num_sample_1,num_sample_2,beta_1,epochs,patience,num_class,temp_path,hidden=128,lr=0.0001,batch_size=128):

    np.random.seed(1)
    random.seed(1)

    c_train = copy.deepcopy(train)
    c_test = copy.deepcopy(test)
    c_valid = copy.deepcopy(valid)
    
  
 
    num_nodes = labels.shape[0]
    features = nn.Embedding(num_nodes, feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    if cuda:
        features.cuda()

    #agg1 = []
    #for i in range(1):
     #   agg1.append(WeightedAggregator(features, cuda=cuda,gcn=True,beta=beta_1,subgraph=True,is_first=True))

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




def train_2l(cuda,gcn,feat_data, labels, adj_lists,adj_lists_new,valid_adjs,train,valid,test,num_sample_1,num_sample_2,beta_1,epochs,patience,num_class,temp_path,hidden=128,lr=0.0001,batch_size=128):

    np.random.seed(1)
    random.seed(1)

    c_train = copy.deepcopy(train)
    c_test = copy.deepcopy(test)
    c_valid = copy.deepcopy(valid)
    
  
 
    num_nodes = labels.shape[0]
    features = nn.Embedding(num_nodes, feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    if cuda:
        features.cuda()

    #agg1 = []
    #for i in range(1):
     #   agg1.append(WeightedAggregator(features, cuda=cuda,gcn=True,beta=beta_1,subgraph=True,is_first=True))

    agg1 = WeightedAggregator_new(features, cuda=cuda,gcn=gcn,beta=beta_1,subgraph=True,is_first=True)
    enc1 = Encoder(features, feat_data.shape[1], hidden, aggregator=agg1,num_sample=num_sample_1, gcn=gcn, cuda=cuda,is_first=True)
    agg2 = WeightedAggregator_new(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(), cuda=cuda,gcn=True,beta=beta_1,subgraph=True,is_first=False)
    enc2 = Encoder(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(), enc1.embed_dim, hidden, aggregator=agg2,num_sample=num_sample_2,
            base_model=enc1, gcn=gcn, cuda=cuda,is_first=False)

    graphsage = SupervisedGraphSage(num_class, enc2)    
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



def train_3l(cuda,gcn,feat_data, labels, adj_lists,adj_lists_new,valid_adjs,train,valid,test,num_sample_1,num_sample_2,beta_1,epochs,patience,num_class,temp_path,hidden=128,lr=0.0001,batch_size=128):

    np.random.seed(1)
    random.seed(1)

    c_train = copy.deepcopy(train)
    c_test = copy.deepcopy(test)
    c_valid = copy.deepcopy(valid)
    
  
 
    num_nodes = labels.shape[0]
    features = nn.Embedding(num_nodes, feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    if cuda:
        features.cuda()

    #agg1 = []
    #for i in range(1):
     #   agg1.append(WeightedAggregator(features, cuda=cuda,gcn=True,beta=beta_1,subgraph=True,is_first=True))

    agg1 = WeightedAggregator_new(features, cuda=cuda,gcn=gcn,beta=beta_1,subgraph=True,is_first=True)
    enc1 = Encoder(features, feat_data.shape[1], hidden, aggregator=agg1,num_sample=num_sample_1, gcn=gcn, cuda=cuda,is_first=True)
    agg2 = WeightedAggregator_new(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(), cuda=cuda,gcn=True,beta=beta_1,subgraph=True,is_first=False)
    enc2 = Encoder(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(), enc1.embed_dim, hidden, aggregator=agg2,num_sample=num_sample_2,
            base_model=enc1, gcn=gcn, cuda=cuda,is_first=False)
    agg3 = WeightedAggregator_new(lambda nodes,adj_lists : enc2(nodes,adj_lists).t(), cuda=cuda,gcn=True,beta=beta_1,subgraph=True,is_first=False)
    enc3 = Encoder(lambda nodes,adj_lists : enc2(nodes,adj_lists).t(), enc2.embed_dim, hidden, aggregator=agg3,num_sample=num_sample_2,
            base_model=enc2, gcn=gcn, cuda=cuda,is_first=False)

    graphsage = SupervisedGraphSage(num_class, enc3)    
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



def train(cuda,gcn,feat_data, labels, adj_lists,adj_lists_new,valid_adjs,train,valid,test,num_sample_1,num_sample_2,beta_1,epochs,patience,num_class,temp_path,hidden=128,lr=0.0001,batch_size=128):

    np.random.seed(1)
    random.seed(1)

    c_train = copy.deepcopy(train)
    c_test = copy.deepcopy(test)
    c_valid = copy.deepcopy(valid)
    
  
 
    num_nodes = labels.shape[0]
    features = nn.Embedding(num_nodes, feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    if cuda:
        features.cuda()

    agg1 = []
    for i in range(3):
        agg1.append(WeightedAggregator(features, cuda=cuda,gcn=True,beta=beta_1,subgraph=True,is_first=True))

    
    enc1 = Encoder(features, feat_data.shape[1], hidden, aggregator=agg1,num_sample=num_sample_1, gcn=gcn, cuda=cuda)
    agg2 = WeightedAggregator(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(), cuda=cuda,gcn=True,beta=beta_1,subgraph=True,is_first=False)
    enc2 = Encoder(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(), enc1.embed_dim, hidden, aggregator=agg2,num_sample=num_sample_2,
            base_model=enc1, gcn=gcn, cuda=cuda)

    graphsage = SupervisedGraphSage(num_class, enc2)    
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
        #print('Epoch: {:04d}'.format(epoch),'loss_train: {:.4f}'.format(np.average(train_losses)),'loss_val: {:.4f}'.format(np.average(valid_losses)),'time: {:.4f}s'.format(time.time() - start_time))
        
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
    #print("Optimization Finished!")
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
        for i in range(10):
            test_output = test_output + graphsage.forward(batch_nodes,adj_lists) 
        
        test_outputs.extend(list(test_output.data.cpu().numpy().argmax(axis=1)))
        test_labels.extend(list(labels[batch_nodes]))
    
    #return accuracy_score(test_labels,test_outputs),f1_score(test_labels,test_outputs,average="micro")
    return [loss_train,loss_values,accuracy_score(test_labels,test_outputs)]



def train_2(feat_data, labels, adj_lists,adj_lists_new,valid_adjs,train,valid,test,num_sample_1,num_sample_2,beta_1,epochs,patience,num_class,temp_path,hidden=128,lr=0.0001,batch_size=128):

    np.random.seed(1)
    random.seed(1)

    c_train = copy.deepcopy(train)
    c_test = copy.deepcopy(test)
    c_valid = copy.deepcopy(valid)
 
    num_nodes = labels.shape[0]
    features = nn.Embedding(num_nodes, feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    
    agg1 = MeanAggregator2(features, cuda=False,gcn=True,beta=beta_1,subgraph=True,is_first=True)
    enc1 = Encoder(features, feat_data.shape[1], hidden, aggregator=agg1,num_sample=num_sample_1, gcn=True, cuda=False)
    agg2 = MeanAggregator2(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(), cuda=False,gcn=True,beta=beta_1,subgraph=True,is_first=False)
    enc2 = Encoder(lambda nodes,adj_lists : enc1(nodes,adj_lists).t(), enc1.embed_dim, hidden, aggregator=agg2,num_sample=num_sample_2,
            base_model=enc1, gcn=True, cuda=False)

    graphsage = SupervisedGraphSage(num_class, enc1)    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=lr)
    
    t_total = time.time()
    loss_values = []
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
            
            loss = graphsage.loss(batch_nodes,valid_adjs,Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            valid_losses.append(loss.data.item())
        print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(np.average(train_losses)),
          'loss_val: {:.4f}'.format(np.average(valid_losses)),
          'time: {:.4f}s'.format(time.time() - start_time))
        
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
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    graphsage.load_state_dict(torch.load(temp_path+'{}.pkl'.format(best_epoch)))        
    print(best_epoch)
    
    num_batches_test = int(len(c_test) / batch_size) 
    test_outputs = []
    test_labels = []
    for batch in range(num_batches_test):
        i_start = batch * batch_size
        i_end = (batch + 1) * batch_size
        batch_nodes = c_test[i_start:i_end]
        test_output = graphsage.forward(batch_nodes,adj_lists) 
        test_outputs.extend(list(test_output.data.cpu().numpy().argmax(axis=1)))
        test_labels.extend(list(labels[batch_nodes]))
    
    return accuracy_score(test_labels,test_outputs),f1_score(test_labels,test_outputs,average="micro")
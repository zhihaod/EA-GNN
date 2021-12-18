from yacs.config import CfgNode as CN

cfg = CN()


def set_cfg(cfg):

    # Set path of results
    cfg.task_path = '/home/zhihao/Document/gnn_fd/graphSage/split/multihead/'

    cfg.dataset = CN()
    # Set path of dataset
    cfg.dataset.path = '/home/zhihao/Document/gnn_fd/graphSage/data/'
    # Set dataset name
    cfg.dataset.name = 'cora'
    # Set dataset split ratio
    cfg.dataset.split = [0.6,0.2,0.2]
    cfg.dataset.task = 'node'
    cfg.dataset.transductive = True

    
    cfg.train = CN()
    cfg.train.batch_size = 128
    cfg.train.patience = 5
    # Set random split rounds
    cfg.train.repeat = 2
    
    cfg.model = CN()
    cfg.model.num_sample = 0
    cfg.model.cuda = True
    # Set start beta
    cfg.model.beta = 0.0
    # Set interval of increasing beta
    cfg.model.interval = 0.2
    # Set search scope: largest beta will be initial beta + interval * (r - 1)
    cfg.model.r = 6
    cfg.model.name = 'ncgnn'
    # Whether concatenation 
    cfg.model.gcn = True
    cfg.model.hidden = 128
    
    
    cfg.optim = CN()
    cfg.optim.lr = 0.001
    # max epochs
    cfg.optim.epochs = 100
    
set_cfg(cfg)


    
    
    
    
    
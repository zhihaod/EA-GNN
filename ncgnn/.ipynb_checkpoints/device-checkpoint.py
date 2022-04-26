import torch
import subprocess
import numpy as np
import pdb
#from graphgym.config import cfg
import os

def get_gpu_memory_map():
    '''Get the current gpu usage.'''
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    return gpu_memory



def auto_select_device(memory_max=8000, memory_bias=200, strategy='random'):
    '''Auto select GPU device'''
    if cuda == True and torch.cuda.is_available():
        
        memory_raw = get_gpu_memory_map()
        if strategy == 'greedy' or np.all(memory_raw > memory_max):
            cuda = np.argmin(memory_raw)
            print('GPU Mem: {}'.format(memory_raw))
            print('Greedy select GPU, select GPU {} with mem: {}'.format(cuda, memory_raw[cuda]))
        elif strategy == 'random':
            memory = 1 / (memory_raw + memory_bias)
            memory[memory_raw > memory_max] = 0
            gpu_prob = memory / memory.sum()
            np.random.seed()
            cuda = np.random.choice(len(gpu_prob), p=gpu_prob)
            np.random.seed(cfg.seed)
            print('GPU Mem: {}'.format(memory_raw))
            print('GPU Prob: {}'.format(gpu_prob.round(2)))
            print(
                'Random select GPU, select GPU {} with mem: {}'.format(
                    cuda, memory_raw[cuda]))
        


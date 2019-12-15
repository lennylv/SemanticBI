import h5py
import pdb
import os

import numpy as np
np.random.seed(0)

from ChIPseq83_preprocess import seq2mat


MPRA_root = '../datasets/Sharpr-MPRA/'

 
def onehot2seq(onehot):
    order = ['A','C','G','T']
    seq = ''
    for l in onehot:
        for i in range(4):
            if l[i]==1:
                seq += order[i]
                break
    return seq
    
def h5py2txt():
    for dataType in ['test','valid','train']:
        if os.path.exists(MPRA_root+dataType+'.txt'):
            continue
        data = h5py.File(MPRA_root+dataType+'.hdf5','r')
        # k562_minp_avg, k562_sv40p_avg, hepg2_minp_avg, hepg2_sv40p_avg
        fw = open(MPRA_root+dataType+'.txt','w')
        for onehot,targets in zip(data['X']['sequence'],data['Y']['output']):
            line = onehot2seq(onehot)+'\t'+' '.join([str(targets[2]),str(targets[5]),str(targets[8]),str(targets[11])])+'\n' 
            fw.write(line)

def txt2data(filePath):
    def line2samples(line):
        seq, targets = line.split('\t')
        onehot = seq2mat(seq)
        # k562 hepg2 minp sv40p
        return [onehot]*4, [[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]], [float(i) for i in targets.split()]
        
    onehots,clprs,targets = [], [], []
    for line in open(filePath):
        if len(line.split('\t')[0])==145:
            onehot4,clpr4,target4 = line2samples(line)
            onehots.extend(onehot4),clprs.extend(clpr4),targets.extend(target4)
    return np.array(onehots),np.array(clprs),np.array(targets)

def loadData_MPRATrain():
    onehots,clprs,targets = txt2data(MPRA_root+'train.txt')
    random_order = np.random.permutation(onehots.shape[0])
    return (onehots[random_order], clprs[random_order]), targets[random_order]

def loadData_MPRAValid():
    onehots,clprs,targets = txt2data(MPRA_root+'valid.txt')
    random_order = np.random.permutation(onehots.shape[0])
    return (onehots[random_order], clprs[random_order]), targets[random_order]

def loadData_MPRATest():
    onehots,clprs,targets = txt2data(MPRA_root+'test.txt')
    l = onehots.shape[0]
    return [[(onehots[range(i,l,4)], clprs[range(i,l,4)]), targets[range(i,l,4)]] for i in range(4)]

if __name__ == "__main__":
    h5py2txt()
    for oo,tc,tt in  loadData_MPRATest():
        pdb.set_trace()

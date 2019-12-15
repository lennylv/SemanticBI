import gzip,csv
import os,pdb
import random
import numpy as np
random.seed(0)

from ChIPseq83_preprocess import base2oenhot_dict, seqs2mats
 
data_root = '../datasets/PBM_66/'
seq_path = data_root+'sequences.tsv.gz'
target_path = data_root+'targets.tsv.gz'


def load_seq():
    return np.array([line[2] for line in list(csv.reader(gzip.open(seq_path,'rt'), delimiter="\t"))[1:]])

def load_target():
    context = list(csv.reader(gzip.open(target_path,'rt'), delimiter="\t"))
    return {tfName:target  for tfName, target in zip(context[0],np.array(context[1:],dtype=float).T)}

def load_data(TF_list):
    
    power_rate = 0.3

    if not set(['TF_%d'%i for i in range(1,67)]).issuperset(TF_list):
        print('Illegal input of TF_list')
        return
    seq, target = load_seq(), load_target()
    TF_list = sorted(TF_list,key=lambda x:int(x.split('TF_')[1]))
    tfDict = {tfName:[0]*len(TF_list) for tfName in TF_list}
    for i, tfName in enumerate(TF_list):
        tfDict[tfName][i]=1

    def data4train(data):
        return [seqs2mats(data[0]), [tfDict[tf] for tf in data[1]]], data[2]

    data_tr, data_te, tr_max, tr_min = {}, {}, {}, {}
    HKME_array = {"ME": range(0, 40524), "HK": range(40524, 80853)}
    for tfName in TF_list:
        tfIndex = int(tfName.split('TF_')[1])
        tr_array, te_array = ['HK','ME'] if tfIndex<=33 else ['ME','HK']
        tr_seq,tr_target = seq[HKME_array[tr_array]],target[tfName][HKME_array[tr_array]]
        tr_nan_mask = ~np.isnan(tr_target)
        tr_seq, tr_target = tr_seq[tr_nan_mask], tr_target[tr_nan_mask]
        trMax, trMin = tr_target.max(), tr_target.min()
        tr_target = np.power((tr_target-trMin)/(trMax-trMin),power_rate)
        tr_max[tfName], tr_min[tfName] = trMax, trMin
        data_tr[tfName] = [tr_seq, [tfName]*len(tr_seq), tr_target]

        te_seq,te_target = seq[HKME_array[te_array]],target[tfName][HKME_array[te_array]]
        te_nan_mask = ~np.isnan(te_target)
        te_seq, te_target = te_seq[te_nan_mask], te_target[te_nan_mask]
        data_te[tfName] = [te_seq, [tfName]*len(te_seq), te_target]
    
    return data_tr, data_te, tr_max, tr_min, power_rate, data4train
 
 
if __name__ == "__main__":
    data_tr, data_te, trMax, trMin, power_rate, data4train = load_data(['TF_%d'%i  for i in range(1,67)])
    pdb.set_trace()

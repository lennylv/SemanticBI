import os,pdb
import gzip,csv
import random
import numpy as np
random.seed(0)

source_root = '../datasets/ChIP-seq_83/'
dist_root = '../datasets/ChIP-seq_83/preprocessed_train/'
dataList_path = '../datasets/ChIP-seq_83/ChIPseq83_list'


base2oenhot_dict = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "C": [0, 0, 1, 0], "G": [0, 0, 0, 1], "N": [0.25, 0.25, 0.25, 0.25]}


def generate_negSample():
    for fh in open(dataList_path):
        fname = fh.strip()+'_AC.seq.gz'
        if fname not in os.listdir(dist_root):
            cmdline = "python DeepRAM_ChIPseq_preprocess.py --ENCODE_data \'%s\' --output \'%s\'"%(source_root+fname,dist_root+fname)
            print(cmdline)
            os.system(cmdline)

def seq2mat(seq):
    return [base2oenhot_dict[base] for base in seq]

def seqs2mats(seqs):
    return [seq2mat(seq) for seq in seqs]

def loadData_ChIPFilePath2data(filePath,tf,cl):
    fin = csv.reader(gzip.open(filePath,'rt'),delimiter='\t')
    fin.__next__()
    return [(line[0], tf, cl, line[1]) for line in fin] if '_AC.' in filePath else [(line[2], tf, cl, line[3]) for line in fin]

def data4model(data):
    fhtfcl, tf2int, cl2int = fileHeadAndtfclDict()
    return (np.array([seq2mat(seq) for seq, tf, cl, label in data]), np.array([tf2int[tf] for seq, tf, cl, label in data]), np.array([cl2int[cl] for seq, tf, cl, label in data])), np.array([int(label) for seq, tf, cl, label in data])

def loadData_ChIPTrain():
    generate_negSample()
    ret = []
    for fileHead, tf, cl in fileHeadAndtfclDict()[0]:
        ret.extend(loadData_ChIPFilePath2data(dist_root+fileHead+'_AC.seq.gz',tf,cl))
    random.shuffle(ret)
    return data4model(ret)

def loadData_ChIPEvaluate():
    for fileHead, tf, cl in fileHeadAndtfclDict()[0]:
        yield fileHead,data4model(loadData_ChIPFilePath2data(dist_root+fileHead+'_AC.seq.gz',tf,cl)),data4model(loadData_ChIPFilePath2data(source_root+fileHead+'_B.seq.gz',tf,cl))

def fileHeadAndtfclDict():
    fileHeadSplit = [[line.strip(),line.split('_')] for line in open(dataList_path)]
    fhtfcl = [[fh,fs[2]+fs[3],fs[1]] if len(fs)==5 else [fh,fs[2],fs[1]] for fh,fs in fileHeadSplit]
    TF_set, CL_set = set(), set()
    for _,TF,CL in fhtfcl:
        TF_set.add(TF), CL_set.add(CL)
    TF_dict = {tf:[0]*len(TF_set) for tf in TF_set}
    CL_dict = {cl:[0]*len(CL_set) for cl in CL_set}
    for i,tf in enumerate(sorted(list(TF_set))):
        TF_dict[tf][i]=1
    for i,cl in enumerate(sorted(list(CL_set))):
        CL_dict[cl][i]=1
    return np.array(fhtfcl),TF_dict,CL_dict

if __name__ == "__main__":
    (seq,tf,cl), label = loadData_ChIPTrain()

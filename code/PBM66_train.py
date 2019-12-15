import os,sys,pdb

import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score
from scipy.stats import pearsonr,spearmanr
from keras.models import Sequential,Model,load_model
from keras.utils.vis_utils import plot_model
from keras.layers.core import Dense, Dropout, Activation,Flatten,Masking,Reshape
from keras.layers.convolutional import Convolution1D,MaxPooling1D,Conv1D
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler,EarlyStopping,TensorBoard
from keras import regularizers
from keras.layers import Input,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers import LSTM,Bidirectional,Input,GRU,Add,Maximum,Concatenate,maximum,GlobalAveragePooling1D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K

from MyCallback import Loss_History
from PBM66_preprocess import load_data

class SemanticBI():
    def __init__(self,modelType='default'):
        self.hyperparam_dict = {k:int(v) if '.' not in v else float(v) for k,v in [line.split(': ') for line in open('../model/%s.param'%modelType)]} 
        self.root = '../model/PBM66_%s/'%(modelType)
        if not os.path.exists(self.root): os.mkdir(self.root)
        self.TF_list =  ['TF_%d'%i  for i in range(1,67)]
        self.data_tr, self.data_te, self.trMax, self.trMin, self.power_rate, self.data4train = load_data(self.TF_list)
        self.model_path = self.root+"best_weight.hdf5"

    def build_model(self):
        dna_input = Input(shape=(40,4),name="seq_input")
        cnn_output = Conv1D(filters=self.hyperparam_dict['cnn_dense'],kernel_size=self.hyperparam_dict['kernel_size'],padding='same',activation="relu",name="cnn_1")(dna_input)
        bn_output = BatchNormalization(name="bn_2")(cnn_output)
        rnn_output = Bidirectional(GRU(units = self.hyperparam_dict['rnn_dense'], return_sequences = True),merge_mode = "sum",name="brnn_1")(bn_output)
        dna_output = GlobalMaxPooling1D()(rnn_output)
        tf_input = Input(shape=(len(self.TF_list),),name="tf_input")
        mlp_input = Concatenate(name="concatenate_1")([dna_output, tf_input])
        mlp_out = Dense(units = self.hyperparam_dict['mlp1_dense'],activation = "relu",name="dense_1")(mlp_input)
        mlp_out = Dropout(rate = self.hyperparam_dict['dropout_1'],name="dropout_1")(mlp_out)
        mlp_out = Dense(units = self.hyperparam_dict['mlp2_dense'],activation = "relu",name="dense_2")(mlp_out)
        mlp_out = Dropout(rate = self.hyperparam_dict['dropout_2'],name="dropout_2")(mlp_out)

        predict = Dense(units = 1,activation="sigmoid",name="sigmoid")(mlp_out)
        self.model = Model(inputs = [dna_input,tf_input],outputs = predict)

        adam=Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss="mse",optimizer=adam,metrics=['mae']) 

    def train_model(self):
        K.set_learning_phase(1)
        self.build_model()
        
        plot_model(self.model,to_file=self.root+'model.png',show_shapes=True)
        checkpoint = ModelCheckpoint(filepath=self.model_path,monitor="val_loss",verbose=0,save_best_only=True)
        reduceLR=ReduceLROnPlateau(monitor='loss',factor=0.1,patience=self.hyperparam_dict['LR_patience'],min_lr=0.00001)
        losshis = Loss_History()
        def scheduler(epoch):
            print ("learning rate:",K.get_value(self.model.optimizer.lr))
            return K.get_value(self.model.optimizer.lr)
        change_lr=LearningRateScheduler(scheduler)
        earlystopping=EarlyStopping(monitor='val_loss',patience=self.hyperparam_dict['ES_patience'],verbose=0)
        
        print ("training model....")
        tr_seq, tr_tf, tr_target = np.concatenate([self.data_tr[key][0] for key in self.data_tr]), np.concatenate([self.data_tr[key][1] for key in self.data_tr]), np.concatenate([self.data_tr[key][2] for key in self.data_tr])
        permu = np.random.permutation(tr_seq.shape[0])
        [tr_seq,tr_tf], tr_target = self.data4train([tr_seq[permu],tr_tf[permu],tr_target[permu]])
        hist = self.model.fit([tr_seq,tr_tf], tr_target, batch_size=self.hyperparam_dict['batchsize'], epochs=100, validation_split=0.1,callbacks=[checkpoint,reduceLR,change_lr,earlystopping,losshis])  
        
        print( "saving model...")
        self.model.save(self.root+"last_weight.hdf5")
        with open(self.root + 'loss.txt',"w") as fout:
            for loss,val_loss in zip(hist.history["loss"],hist.history["val_loss"]):
                fout.write("%.5f\t%.5f\n"%(loss,val_loss))
        losshis.loss_plot(self.root+"loss_curve.png")

    def test_model(self):
        K.set_learning_phase(0)
        print(self.model_path)
        self.model = load_model(self.model_path)
        
        results = []
        for tfName in self.TF_list:
            [te_seq,te_tf], te_target = self.data4train(self.data_te[tfName])
            pred = [p[0] for p in self.model.predict([te_seq,te_tf])]
            pred = np.power(np.maximum(pred,0),1/self.power_rate)*(self.trMax[tfName]-self.trMin[tfName])+self.trMin[tfName]
            label_threshold = np.mean(te_target)+4*np.std(te_target)
            te_label = [0 if i<label_threshold else 1 for i in te_target]
            results.append([tfName,pred,te_target,te_label])
        return results
            

if __name__ == "__main__":    

    def train():

        BI = SemanticBI(modelType='default')
        BI.train_model()
        with open(BI.root+'metrics.txt','w') as fout:
            for tfName, predTE, targetTE, labelTE in BI.test_model():
                line = '%s\t%.4f\t%.4f\t%.4f\n'%(tfName,pearsonr(targetTE,predTE)[0],spearmanr(targetTE,predTE)[0],roc_auc_score(labelTE,predTE))
                print(line)
                fout.write(line)
    train()
import keras
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# https://blog.csdn.net/u013381011/article/details/78911848
class Loss_History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        #self.accuracy.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        #self.val_acc.append(logs.get('val_acc'))

    def loss_plot(self,savepath):
        iters = range(len(self.loss))
        plt.figure()
        plt.plot(iters, self.loss, 'g', label='train loss')
        plt.plot(iters, self.val_loss, 'k', label='val loss')
        #plt.plot(iters, self.accuracy, 'r', label='train acc')
        #plt.plot(iters, self.val_acc, 'b', label='val acc')
        plt.grid(True)
        plt.xlabel("epoch")
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig(savepath,dpi=1000,bbox_inches='tight')
        plt.show()
        plt.close()





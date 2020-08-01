
from IPython.display import clear_output
import tensorflow.keras as keras

import matplotlib.pyplot as plt

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accu = []
        self.val_accu = []
        self.posture_loss = []
        self.val_posture_loss = []
        
        self.torso_accu = []
        self.move_accu = []
        self.val_torso_accu = []
        self.val_move_accu =[]

        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accu.append(logs.get('acc_cust'))
        self.val_accu.append(logs.get('val_acc_cust'))
        self.i += 1
        
        clear_output(wait=True)
        
        # print(self.losses)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.plot(self.x, self.accu, label="acc")
        plt.plot(self.x, self.val_accu, label="val_acc")
        # plt.yscale("log")
        plt.legend()
        plt.show();


from keras import backend as K
import tensorflow as tf

def joint_error_loss(y_true, y_pred, axis):
        return K.sqrt(K.sum(K.square(K.abs(y_true - y_pred)), axis=axis));
    
    
def acc_cust(y_true, y_pred):
    
    y_true_t = tf.reshape(y_true[0], [1, -1, 3])
    y_pred_t = tf.reshape(y_pred[0], [1, -1, 3])
    
    y_pred_t = tf.cast(y_pred_t, tf.float32)
    y_true_t = tf.cast(y_true_t, tf.float32)
   
    joints_err = joint_error_loss(y_pred_t, y_true_t, axis=2);
    correct = K.less(joints_err, 0.100) #tensor with 0 for false values and 1 for true values
    return K.mean(correct) #sum all 1's and divide by the total. 

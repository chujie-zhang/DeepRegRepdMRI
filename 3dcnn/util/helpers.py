
import numpy as np
import nibabel as nib
import os
import pandas as pd
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_data_readers(dir_image0, dir_image1, dir_label0=None, dir_label1=None):

    reader_image0 = DataReader(dir_image0)
    reader_image1 = DataReader(dir_image1)

    reader_label0 = DataReader(dir_label0) if dir_label0 is not None else None
    reader_label1 = DataReader(dir_label1) if dir_label1 is not None else None

    # some checks
    if not (reader_image0.num_data == reader_image1.num_data):
        raise Exception('Unequal num_data between images0 and images1!')
    if dir_label0 is not None:
        if not (reader_image0.num_data == reader_label0.num_data):
            raise Exception('Unequal num_data between images0 and labels0!')
        if not (reader_image0.data_shape == reader_label0.data_shape):
            raise Exception('Unequal data_shape between images0 and labels0!')
    if dir_label1 is not None:
        if not (reader_image1.num_data == reader_label1.num_data):
            raise Exception('Unequal num_data between images1 and labels1!')
        if not (reader_image1.data_shape == reader_label1.data_shape):
            raise Exception('Unequal data_shape between images1 and labels1!')
        if dir_label0 is not None:
            if not (reader_label0.num_labels == reader_label1.num_labels):
                raise Exception('Unequal num_labels between labels0 and labels1!')

    return reader_image0, reader_image1, reader_label0, reader_label1


class DataReader(Dataset):

    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.files = os.listdir(dir_name)
        self.files.sort()
        self.num_data = len(self.files)

        self.file_objects = [nib.load(os.path.join(dir_name, self.files[i])) for i in range(self.num_data)]
        self.num_labels = [self.file_objects[i].shape[3] if len(self.file_objects[i].shape) == 4
                           else 1
                           for i in range(self.num_data)]

        self.data_shape = list(self.file_objects[0].shape[0:3])
        
    def get_num_labels(self, case_indices):
        return [self.num_labels[i] for i in case_indices]
    
    def __getitem(self,index):
        label = np.array([0,0,0,0,-9,-4,-5,-7])
        return np.asarray(self.file_objects[index].dataobj), label[index]
    def __len__(self):
        return self.num_data
    def get_data(self, case_indices=None, label_indices=None):
        if case_indices is None:
            case_indices = range(self.num_data)
        # todo: check the supplied label_indices smaller than num_labels
        if label_indices is None:  # e.g. images only
            data = [np.asarray(self.file_objects[i].dataobj) for i in case_indices]
        else:
            if len(label_indices) == 1:
                label_indices *= self.num_data
            data = [self.file_objects[i].dataobj[..., j] if self.num_labels[i] > 1
                    else np.asarray(self.file_objects[i].dataobj)
                    for (i, j) in zip(case_indices, label_indices)]
        
        #return np.array(data)
        return np.expand_dims(np.stack(data, axis=0), axis=1)


def write_images(input_, file_path=None, file_prefix=''):
    if file_path is not None:
        batch_size = input_.shape[0]
        affine = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]

        [nib.save(nib.Nifti1Image(input_[idx, ...][0,...].to('cpu').numpy(), affine),
                  os.path.join(file_path,
                               file_prefix + '%s.nii' % idx))
         for idx in range(batch_size)]
        

def plot_loss(loss_result_array,epoch,log_number):
    x_axis=np.arange(0,epoch,1).reshape(epoch,1)


    plt.figure(1)
    plt.plot(x_axis,loss_result_array[0,:],label='train_loss',linewidth=1,color='r',marker='o',markerfacecolor='blue',markersize=2) 
    plt.legend()
    plt.plot(x_axis,loss_result_array[1,:],label='validation_loss',linewidth=1,color='g',marker='o',markerfacecolor='blue',markersize=2) 
    plt.legend()
    plt.savefig('loss%s.png'%log_number)
    

    plt.figure(2)
    plt.plot(x_axis,loss_result_array[2,:],label='pos_mse_loss_train',linewidth=1,color='r',marker='o',markerfacecolor='blue',markersize=2) 
    plt.legend()
    plt.plot(x_axis,loss_result_array[3,:],label='neg_mse_loss_train',linewidth=1,color='g',marker='o',markerfacecolor='blue',markersize=2) 
    plt.legend()
    plt.savefig('mse_loss_train%s.png'%log_number)

    
    plt.figure(3)
    plt.plot(x_axis,loss_result_array[4,:],label='pos_new_loss_train',linewidth=1,color='r',marker='o',markerfacecolor='blue',markersize=2) 
    plt.legend()
    plt.plot(x_axis,loss_result_array[5,:],label='neg_new_loss_train',linewidth=1,color='g',marker='o',markerfacecolor='blue',markersize=2) 
    plt.legend()
    plt.savefig('new_loss_train%s.png'%log_number)


    plt.figure(4)
    plt.plot(x_axis,loss_result_array[6,:],label='pos_mse_loss_valid',linewidth=1,color='r',marker='o',markerfacecolor='blue',markersize=2) 
    plt.legend()
    plt.plot(x_axis,loss_result_array[7,:],label='neg_mse_loss_valid',linewidth=1,color='g',marker='o',markerfacecolor='blue',markersize=2) 
    plt.legend()
    plt.savefig('mse_loss_test%s.png'%log_number)

    plt.figure(5)
    plt.plot(x_axis,loss_result_array[8,:],label='pos_new_loss_valid',linewidth=1,color='r',marker='o',markerfacecolor='blue',markersize=2) 
    plt.legend()
    plt.plot(x_axis,loss_result_array[9,:],label='neg_new_loss_valid',linewidth=1,color='g',marker='o',markerfacecolor='blue',markersize=2) 
    plt.legend()
    plt.savefig('new_loss_test%s.png'%log_number)

    plt.show()

def save_loss(loss_result_array,log_number):
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)   
    np.savetxt('loss_result%s'%log_number, loss_result_array, fmt='%.02f')  
    #np.savetxt('loss_result%s.txt'%log_number,np.around(loss_result_array.T,decimals=2))
    
def get_label(pos,neg,train_samples,valid_samples):
    training_label = np.zeros(train_samples)
    test_label = np.zeros(valid_samples)
    for i in range(train_samples):
        if i<train_samples/2:
            training_label[i]=pos
        else:
            training_label[i]=neg
        
    for i in range(valid_samples):
        if i<valid_samples/2:
            test_label[i]=pos
        else:
            test_label[i]=neg
    
    return training_label,test_label

def get_theta_phi(train_path,valid_path,train_samples,valid_samples):
    train = pd.read_csv(train_path,dtype=int)
    test = pd.read_csv(valid_path,dtype=int)
    train = np.array(train)
    test = np.array(test)

    theta_train = np.zeros(train_samples)
    phi_train = np.zeros(train_samples)
    theta_valid = np.zeros(valid_samples)
    phi_valid = np.zeros(valid_samples)
    '''
    theta_train[:] = train[:,0] * -1
    phi_train[:] = train[:,1] * -1

    theta_valid[:] = test[:,0] * -1
    phi_valid[:] = test[:,1] * -1
    '''
    theta_train[train_samples//2:] = train[:,0] * -1
    phi_train[train_samples//2:] = train[:,1] * -1

    theta_valid[valid_samples//2:] = test[:,0] * -1
    phi_valid[valid_samples//2:] = test[:,1] * -1
    
    return theta_train,phi_train,theta_valid,phi_valid
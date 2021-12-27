
import numpy as np
import nibabel as nib
import os
import configparser
import tensorflow as tf

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


class DataReader:

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
        return np.expand_dims(np.stack(data, axis=0), axis=4)


def write_images(input_, file_path=None, file_prefix=''):
    if file_path is not None:
        batch_size = input_.shape[0]
        affine = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        [nib.save(nib.Nifti1Image(input_[idx, ...], affine),
                  os.path.join(file_path,
                               file_prefix + '%s.nii' % idx))
         for idx in range(batch_size)]



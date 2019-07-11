import tensorflow as tf
import os
import numpy as np
import cv2
from scipy.io import loadmat
import pickle
import glob
from util.util_bn import *
from scipy import  misc
p=128

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class DataLoader():
    def __init__(self,train_lst='/home/liang/train.txt',data_dir='/home/liang/',map_parallel_num=8,batch_size=16,shuffle_num=2000,prefetch_num=2000):
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.shuffle_num=shuffle_num
        self.prefetch_num=prefetch_num
        self.train_dir=train_lst
        self.map_parallel_num=map_parallel_num
        self.scale = 8
    def resize(self,prior):
        out = np.zeros((128,128,prior.shape[-1]),dtype=np.uint8)
        for i in range(prior.shape[-1]):
            out[:,:,i] = misc.imresize(prior[:,:,i],1.0,'bicubic')
        return out

    def gen_tfrecords(self,save_dir='tfrecords1',tfrecord_num=10):
        file_num=tfrecord_num
        sample_num=0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fs=[]
        for i in range(file_num):
            fs.append(tf.python_io.TFRecordWriter(os.path.join(save_dir, 'data%d.tfrecords' % i)))
        data_dir = os.listdir('/home/liang/PycharmProjects/facesr/data/train/input')
        
        for img_path in data_dir:
            print(img_path)
            hr = cv2.imread('/home/liang/PycharmProjects/facesr/data/train/input/'+img_path)
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

            landmark = np.load('/home/liang/PycharmProjects/facesr/data/train/input_landmark/'+img_path[:-4]+'.npz')
            landmark = landmark['heatmap']
            landmark = (landmark*255).astype(np.uint8)

            print(landmark.shape,landmark.dtype)

            maps = loadmat('/home/liang/PycharmProjects/facesr/data/train/input_label/'+img_path[:-4]+'.mat')
            maps = maps['pos']
            out_maps = np.zeros((128,128,11),dtype=np.uint8)
            print(out_maps.shape)
            for i in range(1,12):
                out_maps[:,:,i-1] = ((maps==i).astype(np.uint8)*255)
            maps = out_maps
            print(maps.shape,maps.dtype)
            prior = np.concatenate([maps,landmark],axis=-1)
            prior = self.resize(prior)
            print(prior.dtype)
            example = tf.train.Example(features=tf.train.Features(feature={
                'hr': _bytes_feature(hr.tostring()),
                'prior': _bytes_feature(prior.tostring())
            }))
            fs[sample_num % file_num].write(example.SerializeToString())
            sample_num += 1
            if sample_num%1000==0:
                print(sample_num)


        print(sample_num)
        for f in fs:
            f.close()

    def resize(self,prior):
        out = np.zeros((64,64,prior.shape[-1]),dtype=np.uint8)
        for i in range(prior.shape[-1]):
            out[:,:,i] = misc.imresize(prior[:,:,i],0.5,'bicubic')
        return out

    def _parse_one_example(self, example):
        features = tf.parse_single_example(
            example,
            features={
                'hr': tf.FixedLenFeature([], tf.string),
                'prior': tf.FixedLenFeature([], tf.string)
            })
        gt = features['hr']
        gt = tf.decode_raw(gt, tf.uint8)
        gt = tf.reshape(gt, [128,128,3])

        prior = features['prior']
        prior = tf.decode_raw(prior, tf.uint8)
        prior = tf.reshape(prior, [64,64,68+11])

        lr = tf.py_func(lambda x: misc.imresize(x, 1.0 / self.scale, 'bicubic'), [gt], tf.uint8)
        bic = tf.py_func(lambda x: misc.imresize(x, self.scale / 1.0, 'bicubic'), [lr], tf.uint8)

        lr = tf.cast(lr,tf.float32)
        bic = tf.cast(bic,tf.float32)
        gt = tf.cast(gt, tf.float32)
        prior = tf.cast(prior, tf.float32)

        gt = tf.reshape(gt, [p, p, 3]) / 255.0
        bic = tf.reshape(bic, [p, p, 3]) / 255.0
        lr = tf.reshape(lr, [p // self.scale, p // self.scale, 3]) / 255.0
        prior = prior/ 255.0
        return lr, bic, gt , prior

    def read_tfrecords(self, save_dir='tfrecords1'):
        fs_paths = sorted(glob.glob(os.path.join(save_dir, '*.tfrecords')))
        if len(fs_paths) == 0:
            print('No tfrecords. Should run gen_tfrecords() firstly.')
            exit()
        dataset = tf.data.TFRecordDataset(fs_paths)
        print(self.batch_size)
        dataset = dataset.map(self._parse_one_example, self.map_parallel_num).shuffle(self.shuffle_num) \
            .prefetch(self.prefetch_num).batch(self.batch_size).repeat()
        lr, bic, gt , prior = dataset.make_one_shot_iterator().get_next()
        return lr, bic, gt , prior








if __name__=='__main__':
    dataLoader = DataLoader()
    # dataLoader.gen_tfrecords()
    # exit()
    lr, bic, gt , prior =dataLoader.read_tfrecords()
    sess = tf.Session()
    im1,im2,im3,im4=sess.run([lr, bic, gt , prior])
    print(im1.shape,im2.shape,im3.shape,im4.shape)
    for i in range(16):
        a = im2uint8(im1[i])
        b=im2uint8(im2[i])
        c=im2uint8(im3[i])
        d= im2uint8(im4[i][:,:,0])
        e = im2uint8(im4[i][:, :, -1])

        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        b = cv2.cvtColor(b,cv2.COLOR_RGB2BGR)
        c = cv2.cvtColor(c,cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(i) + '_1.png', a)
        cv2.imwrite(str(i)+'_2.png',b)
        cv2.imwrite(str(i)+'_3.png',c)
        cv2.imwrite(str(i) + '_4.png', d)
        cv2.imwrite(str(i) + '_5.png', e)

from __future__ import print_function
import os
import time
import random
import datetime
import scipy.misc
from datetime import datetime
from util.util_bn import *
from data_loader import DataLoader
from skimage.measure import compare_psnr
import cv2
class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = 3
        self.scale = 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels

        # if args.phase == 'train':
        self.crop_size = 256

        self.train_dir = os.path.join('./checkpoints_bn', args.model)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.data_size = 9000 // self.batch_size
        self.max_steps = int(self.epoch * self.data_size)
        self.learning_rate = args.learning_rate

        self.is_training = True




    def generator(self, inputs, reuse=False, scope='g_net'):

        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope('coarseSR'):
            ### coarse SR Network
                x = conv2d(inputs, 'conv1', 64, bn = True, is_training = self.is_training ,activation=True, ksize=3)
                x = resblock(x,64,self.is_training,'res1')
                x = resblock(x, 64, self.is_training, 'res2')
                x = resblock(x, 64, self.is_training, 'res3')
                out1 = conv2d(x, 'conv2', 3)
            with tf.variable_scope('fineSR_encoder'):
                x = conv2d(out1, 'conv1', 64, bn = True, is_training = self.is_training ,activation=True, ksize=3, stride = 2)
                for i in range(12):
                    x = resblock(x, 64, self.is_training,'res'+str(i))
                x = conv2d(x, 'conv2', 64, bn = True, is_training = self.is_training ,activation=True, ksize=3)

            with tf.variable_scope('prior'):
                y = conv2d(out1, 'conv1', 64, bn=True, is_training=self.is_training, activation=True, ksize=7, stride=2)
                for i in range(3):
                    y = resblock(y,128, self.is_training,'res'+str(i))
                y = hour_glass(y,128,4,self.is_training,name='hourglass1')
                y = conv2d(y,'conv2',128, bn=True, is_training=self.is_training, activation=True)
                y = hour_glass(y,128,4,self.is_training,name='hourglass2')
                y1 = conv2d(y,'conv3',68,ksize=1)
                y2 = conv2d(y,'conv4',11,ksize=1)
                y = tf.concat([y1,y2],axis=-1)

            fuse = tf.concat([x,y],axis=-1)

            with tf.variable_scope('fineSR_decoder'):
                x = conv2d(fuse,'conv1',64, bn=True, is_training=self.is_training, activation=True)
                x = deconv2d(x,'deconv1',64,bn=True, is_training=self.is_training, activation=True)
                for i in range(3):
                    x = resblock(x, 64, self.is_training, 'res'+str(i))
                out = conv2d(x,'out',3)

            return out,out1,y


    def build_model(self):

        dataLoader = DataLoader(batch_size=14)

        lr, bic, gt , prior  = dataLoader.read_tfrecords()
        tf.summary.image('bic', im2uint8(bic))
        tf.summary.image('gt', im2uint8(gt))


        # generator
        out,out1,y = self.generator(bic, reuse=False, scope='g_net')

        tf.summary.image('final_out', im2uint8(out))
        tf.summary.image('coarse_out', im2uint8(out1))
        # calculate multi-scale loss
        self.loss_total = 0

        self.coarse_loss = tf.reduce_mean((out1-gt)**2)

        self.prior_loss = tf.reduce_mean((y - prior)**2)

        self.final_loss = tf.reduce_mean((out-gt)**2)


        self.loss_total = self.coarse_loss + self.prior_loss + self.final_loss

        tf.summary.scalar('coarse_loss' , self.coarse_loss)

        tf.summary.scalar('prior_loss', self.prior_loss)

        tf.summary.scalar('final_loss', self.final_loss)

        # losses
        tf.summary.scalar('loss_total', self.loss_total)

        # training vars
        all_vars = tf.trainable_variables()
        self.all_vars = all_vars
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]


        for var in all_vars:
            print(var.name)


    def train(self):
        def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
            train_op = tf.train.RMSPropOptimizer(self.lr)
            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:

                ### if add bn should update mean and var of bn
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = train_op.minimize(loss, global_step, var_list)
            return train_op

        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        self.global_step = global_step

        # build model
        self.build_model()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(2.5e-4, global_step, self.max_steps, end_learning_rate=0.0,
                                            power=0.3)
        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for step in xrange(sess.run(global_step), self.max_steps + 1):

            start_time = time.time()

            # update G network
            _, loss_total_val, coarse_loss_val, prior_loss_val, final_loss_val = sess.run([train_gnet, self.loss_total, self.coarse_loss ,self.prior_loss ,self.final_loss])

            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = (%.5f; %.5f, %.5f,  %.5f)(%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val, coarse_loss_val, prior_loss_val, final_loss_val ,
                                     examples_per_sec, sec_per_batch))

            if step % 20 == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save the model checkpoint periodically.
            if step % 500 == 0 or step == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, step)

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test(self, height, width, input_path, output_path):
        self.is_training = False
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = sorted(os.listdir(input_path))

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        img_sr, local_sr, prior = self.generator(inputs, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()

        best_psnr = 0.0
        best_step = -1

        for step in range(37000,37500,500):
            self.load(sess, os.path.join(self.train_dir,'checkpoints'), step=step)
            avg_psnr = 0.0
            for imgName in imgsName:
                blur = scipy.misc.imread(os.path.join(input_path, imgName))

                lr = scipy.misc.imresize(blur,0.125,'bicubic')
                bic = scipy.misc.imresize(lr, 8.0, 'bicubic')

                blurPad = np.expand_dims(bic, 0)

                start = time.time()
                res = sess.run(img_sr, feed_dict={inputs: blurPad / 255.0})
                duration = time.time() - start

                res = im2uint8(res[0, :, :, :])
                avg_psnr += compare_psnr(res,blur)
                res = cv2.cvtColor(res,cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_path,imgName),res)
            avg_psnr /= len(imgsName)
            print(step,avg_psnr)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_step = step


        print(best_psnr,best_step)


                # scipy.misc.imsave(os.path.join(output_path, imgName), res)

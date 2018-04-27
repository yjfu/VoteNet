from network_config import NetworkConfig
from deepLab_v2 import DeepLabV2
from dataset import Dataset
from utils.convert_label import inv_preprocess, decode_labels
import tensorflow as tf
import numpy as np
import time
import os
class Model_DLV2:
    def __init__(self):
        self.config = NetworkConfig()
        self.coord = tf.train.Coordinator()
    def train(self):
        self._train_setup()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if self.config.pretrain_file is not None:
                self.loader.restore(sess, self.config.pretrain_file)
                print("Restored model parameters from {}".format(self.config.pretrain_file))
                # Start queue threads.
                threads = tf.train.start_queue_runners(coord=self.coord, sess=sess)

                # Train!
                for step in range(self.config.num_steps + 1):
                    start_time = time.time()
                    feed_dict = {self.curr_step: step}

                    loss_value, summary, mat,le, _ = sess.run(
                        [self.reduced_loss,
                         self.summary,
                         self.confusion_matrix,
                         self.loss_e,
                         self.train_op],
                        feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary, step)
                    if step % self.config.save_interval == 0:
                        self._train_save(self.saver, step, sess)

                    duration = time.time() - start_time
                    print('step {:d} \t loss = {:.3f}, le = {:.3f}, ({:.3f} sec/step)'
                          .format(step, loss_value, le, duration))
                    self.compute_IoU_per_class(mat)

                # finish
                self.coord.request_stop()
                self.coord.join(threads)

    def _train_setup(self):
        self.dataset = Dataset(index_path=self.config.data_list,
                               mode='train',
                               index_base=self.config.data_dir,
                               random_scale=self.config.random_scale,
                               random_mirror=self.config.random_mirror,
                               ignore_label=self.config.ignore_label,
                               img_mean=self.config.IMG_MEAN,
                               img_size=(self.config.input_height,
                                         self.config.input_width))
        self.images, self.masks = self.dataset.dequeue(self.config.batch_size)
        self.net = DeepLabV2(self.images, self.config.num_classes)
        self.net_output = self.net.output
        self._train_def_variables_group()
        self._train_def_losses2()
        self._train_def_train_op()
        self._train_summary_accuracy()
        # Saver for storing checkpoints of the model
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)

        # Loader for loading the pre-trained model
        self.loader = tf.train.Saver(var_list=self.restore_var)

    def _train_def_variables_group(self):
        """
        split variables to several group
        restore_var: all variables(no matter it is trainable or not)except those in
                    decoder(ASPP). They will be restore from pre-trained models
        all_trainable: all variables which are trainable, note that because in fact
                    we set trainable==false when using batch normal layers, so gamma
                    and beta is not in here but frozen since restored
        encoder_trainable: trainable variables in encoder, i.e trainable variables
                    in restore_var, which desire a small learning rate
        decoder_trainable: trainable variables in decoder, i.e trainable variables
                    not in restore_var, which desire a greater learning rate
        decoder_w_trainable: weights in filters in decoder_trainable
        decoder_w_trainable: weights in bias in decoder_trainable
        :return:
        """
        # for restore
        self.restore_var = [v for v in tf.global_variables() if 'fc' not in v.name]
        # for training use different learning rate
        self.all_trainable = tf.trainable_variables()
        # Fine-tune part
        self.encoder_trainable = [v for v in self.all_trainable if 'fc' not in v.name]  # lr * 1.0
        # Decoder part
        self.decoder_trainable = [v for v in self.all_trainable if 'fc' in v.name]

        self.decoder_w_trainable = [v for v in self.decoder_trainable
                                    if 'weights' in v.name or 'gamma' in v.name]  # lr * 10.0
        self.decoder_b_trainable = [v for v in self.decoder_trainable
                                    if 'biases' in v.name or 'beta' in v.name]  # lr * 20.0
        # Check
        assert (len(self.all_trainable) == len(self.decoder_trainable)
                + len(self.encoder_trainable))
        assert (len(self.decoder_trainable) == len(self.decoder_w_trainable)
                + len(self.decoder_b_trainable))

    def _train_def_losses(self):
        """
        softmax_cross_entropy
        :return:
        """
        # Output size
        output_shape = tf.shape(self.net_output)
        self.output_size = (output_shape[1], output_shape[2])

        # Groud Truth: ignoring all labels greater or equal than n_classes
        label_proc = self._resize_label(self.masks, self.output_size)
        raw_gt = tf.reshape(label_proc, [-1, ])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.config.num_classes - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        raw_prediction = tf.reshape(self.net_output, [-1, self.config.num_classes])
        prediction = tf.gather(raw_prediction, indices)

        # Pixel-wise softmax_cross_entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        # L2 regularization
        l2_losses = [self.config.weight_decay * tf.nn.l2_loss(v) for v in self.all_trainable if 'weights' in v.name]
        # Loss function
        self.reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
        tf.summary.scalar("loss", self.reduced_loss)

    def _train_def_losses2(self):
        # Output size
        output_shape = tf.shape(self.net_output)
        self.output_size = (output_shape[1], output_shape[2])

        # Groud Truth: ignoring all labels greater or equal than n_classes
        label_proc = self._resize_label(self.masks, self.output_size)
        raw_gt = tf.reshape(label_proc, [-1, ])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.config.num_classes - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        raw_prediction = tf.reshape(self.net_output, [-1, self.config.num_classes])
        prediction = tf.gather(raw_prediction, indices)


        one_hot_gt = tf.cast(tf.one_hot(gt, depth=self.config.num_classes), tf.float32)

        pre_res = tf.nn.softmax(prediction)
        pre_likely = tf.cast(tf.greater(pre_res, 1.0/self.config.num_classes), tf.float32)

        sum = tf.reduce_sum(tf.multiply(pre_res, pre_likely), axis=1)
        correct = tf.reduce_sum(tf.multiply(one_hot_gt, pre_res), axis=1)
        weight = tf.subtract(sum, correct)
        print "new weight"
        loss_e = one_hot_gt*tf.log((one_hot_gt+(1-one_hot_gt)*(1e-10))/(pre_res+(1e-10)))
        loss_e = tf.reduce_sum(loss_e, axis=1)
        loss_e = tf.multiply(weight, loss_e)
        print "loss_e: ", loss_e.shape
        print "weight: ", weight.shape
        print "sum: ", sum.shape
        print "correct: ", correct.shape
        print "pre_res: ", pre_res.shape
        #print "onehot: ", one_hot_gt.shape



        # Pixel-wise softmax_cross_entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        # L2 regularization
        l2_losses = [self.config.weight_decay * tf.nn.l2_loss(v) for v in self.all_trainable if 'weights' in v.name]
        # Loss function
        print "loss: ", loss.shape
        self.reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
        self.reduced_loss2 = tf.reduce_mean(loss_e) + tf.add_n(l2_losses)
        self.loss_e = tf.reduce_mean(loss_e) + tf.add_n(l2_losses)
        tf.summary.scalar("loss", self.reduced_loss)
        tf.summary.scalar("loss_e", tf.reduce_mean(loss_e)*10 + tf.add_n(l2_losses))

    def _train_def_train_op(self):
        """
        apply different learning rate for different group of variables
        :return:
        """
        # Define optimizers
        # 'poly' learning rate
        base_lr = tf.constant(self.config.learning_rate)
        self.curr_step = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - self.curr_step / self.config.num_steps), self.config.power))

        opt_encoder = tf.train.MomentumOptimizer(learning_rate, self.config.momentum)
        opt_decoder_w = tf.train.MomentumOptimizer(learning_rate * 10.0, self.config.momentum)
        opt_decoder_b = tf.train.MomentumOptimizer(learning_rate * 20.0, self.config.momentum)
        # To make sure each layer gets updated by different lr's, we do not use 'minimize' here.
        # Instead, we separate the steps compute_grads+update_params.
        # Compute grads
        grads = tf.gradients((self.reduced_loss2+self.reduced_loss), self.encoder_trainable + self.decoder_w_trainable + self.decoder_b_trainable)
        print "loss 2.1"
        grads_encoder = grads[:len(self.encoder_trainable)]
        grads_decoder_w = grads[len(self.encoder_trainable): (len(self.encoder_trainable) + len(self.decoder_w_trainable))]
        grads_decoder_b = grads[(len(self.encoder_trainable) + len(self.decoder_w_trainable)):]
        # Update params
        train_op_conv = opt_encoder.apply_gradients(zip(grads_encoder, self.encoder_trainable))
        train_op_fc_w = opt_decoder_w.apply_gradients(zip(grads_decoder_w, self.decoder_w_trainable))
        train_op_fc_b = opt_decoder_b.apply_gradients(zip(grads_decoder_b, self.decoder_b_trainable))
        # Finally, get the train_op!
        # for collecting moving_mean and moving_variance for batch normal
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)


    def _train_summary_accuracy(self):
        """
        get confusion_matrix to calculate mIoU of each batch when training
        :return:
        """
        raw_output = tf.image.resize_bilinear(self.net_output, tf.shape(self.images)[1:3, ])
        raw_output = tf.argmax(raw_output, axis=3)
        pred = tf.expand_dims(raw_output, dim=3)
        self.pred = tf.reshape(pred, [-1, ])
        # labels
        gt = tf.reshape(self.masks, [-1, ])
        # Ignoring all labels greater than or equal to n_classes.
        temp = tf.less_equal(gt, self.config.num_classes - 1)
        weights = tf.cast(temp, tf.int32)

        # fix for tf 1.3.0
        gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))
        # confusion matrix
        self.confusion_matrix = tf.contrib.metrics.confusion_matrix(
            self.pred, gt, num_classes=self.config.num_classes, weights=weights)

        self.summary = tf.summary.merge_all()
        if not os.path.exists(self.config.logdir):
            os.makedirs(self.config.logdir)
        self.summary_writer = tf.summary.FileWriter(self.config.logdir, graph=tf.get_default_graph())

    def _train_save(self, saver, step, sess):
        '''
        Save weights.
        '''
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(self.config.modeldir, model_name)
        if not os.path.exists(self.config.modeldir):
            os.makedirs(self.config.modeldir)
        saver.save(sess, checkpoint_path, global_step=step)
        print('The checkpoint has been created.')

    def _resize_label(self, input_batch, new_size):
        """
        Resize masks
        Args:
          input_batch: input tensor of shape [batch_size H W 1].
          new_size: a tensor with new height and width.
          num_classes: number of classes to predict (including background).
        Returns:
        """
        with tf.name_scope('label_encode'):
            # as labels are integer numbers, need to use NN interp.
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size)
            # reducing the channel dimension.
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3])
        return input_batch

    def compute_IoU_per_class(self, confusion_matrix):
        mIoU = 0
        class_num = self.config.num_classes
        for i in range(self.config.num_classes):
            # IoU = true_positive / (true_positive + false_positive + false_negative)
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i]) - TP
            if (TP + FP + FN) == 0:
                class_num -= 1
                continue
            IoU = TP*1.0 / (TP + FP + FN)
            # print ('class %d: %.3f' % (i, IoU))
            mIoU += IoU
        mIoU /= class_num
        print ('mIoU: %.3f' % mIoU)
        return mIoU

    def test(self):
        self._test_setup()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # load checkpoint
            checkpointfile = self.config.modeldir + '/model.ckpt-' + str(self.config.valid_step)
            self.loader.restore(sess, checkpointfile)
            print("Restored model parameters from {}".format(checkpointfile))

            # Start queue threads.
            threads = tf.train.start_queue_runners(coord=self.coord, sess=sess)

            # Test!
            confusion_matrix = np.zeros((self.config.num_classes, self.config.num_classes), dtype=np.int)
            for step in range(self.config.valid_num_steps):
                preds, _, _, c_matrix = sess.run(
                    [self.pred, self.accu_update_op, self.mIou_update_op, self.confusion_matrix])
                confusion_matrix += c_matrix
                if step % 100 == 0:
                    print('step {:d}'.format(step))
            print('Pixel Accuracy: {:.3f}'.format(self.accu.eval(session=sess)))
            print('Mean IoU: {:.3f}'.format(self.mIoU.eval(session=sess)))
            self.compute_IoU_per_class(confusion_matrix)

            # finish
            self.coord.request_stop()
            self.coord.join(threads)
    def _test_setup(self):
        # Create queue coordinator.
        self.coord = tf.train.Coordinator()
        self.dataset = Dataset(index_path=self.config.test_data_list,
                               mode='test',
                               index_base=self.config.data_dir,
                               random_scale=False,
                               random_mirror=False,
                               ignore_label=self.config.ignore_label,
                               img_mean=self.config.IMG_MEAN,
                               img_size=None)
        self.images, self.masks = self.dataset.img, self.dataset.mask  # [h, w, 3 or 1]
        # Add one batch dimension [1, h, w, 3 or 1]
        self.images, self.masks = tf.expand_dims(self.images, dim=0), tf.expand_dims(self.masks, dim=0)

        self.net = DeepLabV2(self.images, self.config.num_classes)

        # predictions
        raw_output = self.net.output
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(self.images)[1:3, ])
        raw_output = tf.argmax(raw_output, axis=3)
        pred = tf.expand_dims(raw_output, dim=3)
        self.pred = tf.reshape(pred, [-1, ])
        # labels
        gt = tf.reshape(self.masks, [-1, ])
        # Ignoring all labels greater than or equal to n_classes.
        temp = tf.less_equal(gt, self.config.num_classes - 1)
        weights = tf.cast(temp, tf.int32)

        # fix for tf 1.3.0
        gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))

        # Pixel accuracy
        self.accu, self.accu_update_op = tf.contrib.metrics.streaming_accuracy(
            self.pred, gt, weights=weights)

        # mIoU
        self.mIoU, self.mIou_update_op = tf.contrib.metrics.streaming_mean_iou(
            self.pred, gt, num_classes=self.config.num_classes, weights=weights)

        # confusion matrix
        self.confusion_matrix = tf.contrib.metrics.confusion_matrix(
            self.pred, gt, num_classes=self.config.num_classes, weights=weights)

        # Loader for loading the checkpoint
        self.loader = tf.train.Saver(var_list=tf.global_variables())
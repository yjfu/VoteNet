import os
import tensorflow as tf
class Dataset:

    def __init__(self, index_path, mode, index_base, random_scale, random_mirror,
                 ignore_label, img_mean, img_size=None,):
        """
        init the dataset from an index file
        :param index_path: index file must be of form 'path/to/img path/to/mask'
        :param mode: 'train' or 'test'
        :param index_base: base path of data index,so the path should be
                    'index_base/path/to/img'
        :param random_scale: whether do random scale or not when training
        :param random_mirror: whether do random mirror or not when training
        :param ignore_label: which kind of label should be ignored
        :param img_mean: to be subtract by img, pixel-wise
        :param img_size: all img will be re-sized to this size when training
        """
        self.index_path = index_path
        self.mode = mode
        self.index_base = index_base
        self.img_size = img_size
        self._read_index_file()
        self.queue = tf.train.slice_input_producer([self.img_paths, self.mask_paths],
                                                   shuffle=('train' == self.mode))
        self._define_read_img_and_preprocess(random_scale, random_mirror, ignore_label,
                                             img_mean)


    def _read_index_file(self):
        """
        to form data list
        :return:
        """
        lines = open(self.index_path)
        self.img_paths = []
        self.mask_paths = []
        for line in lines:
            img, mask = line.split()
            img = os.path.join(self.index_base, img)
            mask = os.path.join(self.index_base, mask)
            self.img_paths.append(img)
            self.mask_paths.append(mask)

        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.mask_paths = tf.convert_to_tensor(self.mask_paths, dtype=tf.string)

    def _image_scaling(self, img, label):
        """
        Randomly scales the images between 0.5 to 1.5 times the original size.

        Args:
          img: Training image to scale.
          label: Segmentation mask to scale.
        """

        scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
        h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
        w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
        new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
        img = tf.image.resize_images(img, new_shape)
        label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
        label = tf.squeeze(label, squeeze_dims=[0])

        return img, label

    def _image_mirroring(self, img, label):
        """
        Randomly mirrors the images.

        Args:
          img: Training image to mirror.
          label: Segmentation mask to mirror.
        """

        distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
        mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
        mirror = tf.boolean_mask([0, 1, 2], mirror)
        img = tf.reverse(img, mirror)
        label = tf.reverse(label, mirror)
        return img, label

    def _random_crop_and_pad_image_and_labels(self, image, label, crop_h, crop_w, ignore_label=255):
        """
        Randomly crop and pads the input images.

        Args:
          image: Training image to crop/ pad.
          label: Segmentation mask to crop/ pad.
          crop_h: Height of cropped segment.
          crop_w: Width of cropped segment.
          ignore_label: Label to ignore during the training.
        """

        label = tf.cast(label, dtype=tf.float32)
        label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.
        combined = tf.concat(axis=2, values=[image, label])
        image_shape = tf.shape(image)
        combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
                                                    tf.maximum(crop_w, image_shape[1]))

        last_image_dim = tf.shape(image)[-1]
        # last_label_dim = tf.shape(label)[-1]
        combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 4])
        img_crop = combined_crop[:, :, :last_image_dim]
        label_crop = combined_crop[:, :, last_image_dim:]
        label_crop = label_crop + ignore_label
        label_crop = tf.cast(label_crop, dtype=tf.uint8)

        # Set static shape so that tensorflow knows shape at compile time.
        img_crop.set_shape((crop_h, crop_w, 3))
        label_crop.set_shape((crop_h, crop_w, 1))
        return img_crop, label_crop

    def _define_read_img_and_preprocess(self, random_scale, random_mirror,
                                        ignore_label, img_mean):
        """
        read the first pair of img and mask from self.queue
        :return:
        """
        img = tf.read_file(self.queue[0])
        mask = tf.read_file(self.queue[1])

        img = tf.image.decode_jpeg(img)
        mask = tf.image.decode_png(mask, channels=1)

        img = tf.cast(img, dtype=tf.float32)
        img -= img_mean
        if self.mode == 'train':
            h, w = self.img_size
            if random_scale:
                img, mask = self._image_scaling(img, mask)
            if random_mirror:
                img, mask = self._image_mirroring(img, mask)
            img, mask = self._random_crop_and_pad_image_and_labels(img, mask, h, w, ignore_label)
        self.img = img
        self.mask = mask

    def dequeue(self, batch_size):
        img_batch, mask_batch= tf.train.batch([self.img, self.mask], batch_size=batch_size)
        return img_batch, mask_batch


# index_path = os.path.join(os.path.dirname(__file__), "data",
#                            "train.txt")
# index_base = os.path.join(os.path.dirname(__file__), "..")
# d = Dataset(index_path, 'train', index_base)
# img, mask = d.dequeue()
# with tf.Session() as sess:
#     coor = tf.train.Coordinator()
#     thread = tf.train.start_queue_runners(sess, coor)
#     for i in range(3):
#         print i
#         ii, mm = sess.run([img, mask])
#         print ii, mm
#     coor.request_stop()
#     coor.join(thread)
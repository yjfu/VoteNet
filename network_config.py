import numpy as np

class NetworkConfig:
    def __init__(self):
        self._def_train_para()
        self._def_validation_para()
        self._saving_output_setting()
        self._data_setting()
        self._log_setting()

    def _def_train_para(self):
        self.num_steps = 20000                              # maximum number of iterations
        self.save_interval = 1000                           # number of iterations for saving and visualization
        self.random_seed = 1234                             # random seed
        self.weight_decay = 0.0005                          # weight decay rate
        self.learning_rate = 2.5e-4                         # learning rate
        self.power = 0.9                                    # hyperparameter for poly learning rate
        self.momentum = 0.9                                 # momentum
        self.encoder_name = 'deeplab'                       # name of pre-trained model, res101, res50 or deeplab
        self.pretrain_file = './model/deeplab_init/deeplab_resnet_init.ckpt'
                                                            # pre-trained model filename corresponding to encoder_name
        self.data_list = './data/train.txt'              # training data list filename
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    def _def_validation_para(self):
        self.valid_step = 20000                             # checkpoint number for validation
        self.valid_num_steps = 1449                         # number of validation samples
        self.valid_data_list = './data/val.txt'          # validation data list filename

    def _saving_output_setting(self):
        self.out_dir = 'output'                             # directory for saving outputs
        self.test_step = 20000                              # checkpoint number for testing/validation
        self.test_num_steps = 1449                          # number of testing/validation samples
        self.test_data_list = './data/train.txt'           # testing/validation data list filename
        self.visual = True                                  # whether to save predictions for visualization

    def _data_setting(self):
        self.data_dir = '../'        # data directory
        self.batch_size = 4                                # training batch size
        self.input_height = 321                             # input image height
        self.input_width = 321                              # input image width
        self.num_classes = 21                               # number of classes
        self.ignore_label = 255                             # label pixel value that should be ignored
        self.random_scale = True                            # whether to perform random scaling data-augmentation
        self.random_mirror = True                           # whether to perform random flipping data-augmentation

    def _log_setting(self):
        self.modeldir = 'model/new3'                             # model directory
        self.logfile = 'log.txt'                            # training log filename
        self.logdir = 'log'                                 # training log directory

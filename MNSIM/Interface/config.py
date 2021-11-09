from easydict import EasyDict as edict
import os.path

this_path = os.path.dirname(os.path.realpath(__file__))

cfg = edict()

cfg.MODE = 'FIX_TRAIN'

cfg.TRAIN_BATCH_SIZE = 32
cfg.TRAIN_NUM_WORKERS =5
cfg.TEST_BATCH_SIZE = 64
cfg.TEST_NUM_WORKERS = 5

cfg.voc = edict()
cfg.voc.TRAIN_SIZE = [416, 416]
cfg.voc.VAL_SIZE = [416, 416]
cfg.voc.num_classes = 20
cfg.voc.root = this_path + "/data/voc2007/VOCdevkit"
# training parameters
cfg.train = edict()
cfg.train.MOMENTUM = 0.9
cfg.train.WEIGHT_DECAY = 0.0005
cfg.train.GAMMA = 0.01
cfg.train.lr = 0.0001
cfg.train.MILESTONES = [30, 60]
cfg.train.EPOCHS = 90
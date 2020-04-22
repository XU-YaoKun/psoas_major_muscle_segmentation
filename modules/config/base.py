from yacs.config import CfgNode as CN

_C = CN()

# public alias
cfg = _C

# Model
_C.MODEL = CN()
_C.MODEL.TYPE = "FCN"
_C.MODEL.N_CHANNELS = 1
_C.MODEL.N_CLASS = 1

# Data
_C.DATA = CN()
_C.DATA.DATA_PATH = ""
_C.DATA.BATCH_SIZE = 4
_C.DATA.NUM_WORKERS = 8
_C.DATA.TEST_PERCENT = 0.2

# Train
_C.TRAIN = CN()
_C.TRAIN.LR = 1e-3
_C.TRAIN.WEIGHT_DECAY = 1e-8
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.EPOCH = 50

# Misc
_C.OUTPUT_DIR = "outputs"
_C.TEST_STEP = 3

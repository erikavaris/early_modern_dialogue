import os

class Configuration(object):
    DATA_DIR = os.getenv('DATA_DIR', 'data') #replace 'data' with your data directory
    VOCAB_SIZE = os.getenv('VOCAB_SIZE', 55000)
    SIZE = os.getenv('SIZE', 1024)
    NUM_LAYERS = os.getenv('NUM_LAYERS', 3)
    MAX_GRADIENT_NORM = os.getenv('MAX_GRADIENT_NORM', 5.0)
    BATCH_SIZE = os.getenv('BATCH_SIZE', 64)
    LEARNING_RATE = os.getenv('LEARNING_RATE', .5)
    LEARNING_RATE_DECAY_FACTOR = os.getenv('LEARNING_RATE_DECAY_FACTOR', .99)
    TRAIN_DIR = os.getenv('TRAIN_DIR', 'checkpoints')

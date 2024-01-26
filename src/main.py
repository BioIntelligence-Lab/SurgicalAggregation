import argparse

import tensorflow as tf
from train import *
from test import test_models
from analysis import analyze_models

parser = argparse.ArgumentParser()
parser.add_argument('-train_baseline', action='store_true')
parser.add_argument('-train_simulation', action='store_true')
parser.add_argument('-train_simulation_exp1', action='store_true')
parser.add_argument('-train_simulation_exp2', action='store_true')
parser.add_argument('-train_realworld', action='store_true')
parser.add_argument('-test', action='store_true')
parser.add_argument('-analyze', action='store_true')
args = parser.parse_args()

if __name__=='__main__':
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)
  # Run experiment based on passed arguments   
  if args.train_baseline:
    train_baseline()
  if args.train_simulation:
    train_simulation_experiments()
  if args.train_simulation_exp1:
    train_simulation_experiments(['exp1.1'])
  if args.train_simulation_exp2:
    train_simulation_experiments(['exp1.2'])
  if args.train_realworld:
    train_realworld_experiments()
  if args.test:
    test_models()
  if args.analyze:
    analyze_models()
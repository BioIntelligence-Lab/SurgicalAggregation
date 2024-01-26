'''
Configures model architecture
'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers

from dataset import Dataset
from utils import *

'''
Centralized/local
'''


def __train_local(
  train_ds,
  val_ds,
  ckpt_dir,
  ckpt_name = 'model.hdf5',
  learning_rate = 1e-3,
  warmup_epochs = 15,
  epochs = 1000,
  image_shape = (224,224,3),
  only_warmup_model = False,
  early_stopping = True
):
  # Sanity check before training!
  if train_ds.labels != val_ds.labels:
    raise ValueError('Mismatched labels!')
  # Get info
  labels = train_ds.labels
  train_data = train_ds.get_dataset(image_shape[:2])
  val_data = val_ds.get_dataset(image_shape[:2])
  # Initialize model
  model, base_model = create_model(len(labels), image_shape)
  model.compile(
    optimizer = keras.optimizers.Adam(learning_rate),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = keras.metrics.AUC(curve='ROC', name='auc', multi_label=True)
  )
  os.makedirs(os.path.join(MODEL_DIR, ckpt_dir), exist_ok=True)
  # Save model checkpoints based on validation loss
  checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, ckpt_dir, ckpt_name),
    monitor = 'val_loss',
    mode = 'min',
    save_best_only = True
  )
  # Warmup model classification head
  model.fit(
    train_data, 
    validation_data = val_data,
    epochs = warmup_epochs, 
    callbacks = [checkpoint],
    use_multiprocessing = True
  )
  if not only_warmup_model:
    os.makedirs(os.path.join(LOGS_DIR, ckpt_dir), exist_ok=True)
    # Load "best" model so far
    model.load_weights(os.path.join(MODEL_DIR, ckpt_dir, ckpt_name))
    # Unfreeze all base models layers except BatchNormalization layers
    vars = len(model.trainable_variables)
    base_model.trainable = True
    for layer in model.layers:
      if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
    # Sanity check!
    print('Trainable Variables: ' + str(vars) + ' -> ' + str(len(model.trainable_variables)))
    # Recompile with slower learning rate
    model.compile(
      optimizer = keras.optimizers.Adam(5e-2*learning_rate),
      loss = keras.losses.BinaryCrossentropy(),
      metrics = keras.metrics.AUC(curve='ROC', name='auc', multi_label=True)
    )
    callbacks = [checkpoint]
    # Early stopping 
    if early_stopping:
      callbacks += [
        tf.keras.callbacks.EarlyStopping(
          monitor = 'val_loss', 
          patience = 25
        )
      ]
    # Fine-tune all trainable layers
    logs = model.fit(
      train_data, 
      validation_data = val_data,
      epochs = epochs, 
      callbacks = callbacks,
      use_multiprocessing = True
    )
    logs = pd.DataFrame(logs.history)
    logs['epoch'] = np.arange(logs.shape[0])
    logs = logs[['epoch', 'loss', 'auc', 'val_loss', 'val_auc']]
    logs.to_csv(os.path.join(LOGS_DIR, ckpt_dir, f'{ckpt_name[:-5]}_logs.csv'), index=False)

def train_baseline(
  train_ds,
  val_ds,
  ckpt_dir, 
  learning_rate = 1e-3,
  warmup_epochs = 15,
  epochs = 1000,
  image_shape = (224,224,3),
  early_stopping = True
):
  return __train_local(train_ds, val_ds, ckpt_dir, 'model.hdf5', learning_rate,warmup_epochs, epochs, image_shape, early_stopping = early_stopping)
  
def train_centralized(
  train_ds,
  val_ds,
  ckpt_dir, 
  learning_rate = 1e-3,
  warmup_epochs = 15,
  epochs = 1000,
  image_shape = (224,224,3),
  early_stopping = True
):
  print(f'Central Aggregation')
  for i in range(len(train_ds)):
    print(f'  # of labels for client {i+1} = {len(train_ds[i].labels)}')
  train_ds = Dataset.merge(train_ds)
  val_ds = Dataset.merge(val_ds)
  print(f'# of labels for central aggregation = {len(train_ds.labels)}')
  return __train_local(train_ds, val_ds, ckpt_dir, 'model.hdf5', learning_rate, warmup_epochs, epochs, image_shape, early_stopping = early_stopping)
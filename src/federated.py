'''
Configures model architecture
'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, backend
from enum import Enum

from dataset import union_labels
from local import __train_local
from utils import *

'''
Federated/Distributed
'''

class Strategies(Enum):
  FedAvg = 0
  FedBN = 1
  FedBNplus = 2

def __get_local_weights(K, local_models, layer_idx):
  local_weights = []
  for k in range(K):
    local_weights += [local_models[k].layers[layer_idx].get_weights()]
  return np.array(local_weights, dtype=object)

def __get_new_weights(local_weights):
  # Calculate mean of each trainable parameter in layer
  new_weights = []
  for w in zip(*local_weights):
    new_weights += [np.mean(w, axis=0)]
  return new_weights

def __fedavg(K, global_model, local_models, layer_idx):
  '''Federated Averaging (FedAvg)'''
  # Fetch weights for each layer from all clients
  local_weights = __get_local_weights(K, local_models, layer_idx)
  # Aggregate weights
  new_weights = __get_new_weights(local_weights)
  # Update weights for global model
  global_model.layers[layer_idx].set_weights(new_weights)
  # Sync aggregated weights across all clients
  for k in range(K):
    local_models[k].layers[layer_idx].set_weights(new_weights)
  return global_model, local_models

def __fedbn(K, global_model, local_models, layer_idx):
  '''FedBN. Modification of FedAvg'''
  if type(global_model.layers[layer_idx]).__name__ != 'BatchNormalization':
    # Fetch weights for each layer from all clients
    local_weights = __get_local_weights(K, local_models, layer_idx)
    # Aggregate weights
    new_weights = __get_new_weights(local_weights)
    # Update weights for global model
    global_model.layers[layer_idx].set_weights(new_weights)
    # Sync aggregated weights across all clients
    for k in range(K):
      local_models[k].layers[layer_idx].set_weights(new_weights)
  return global_model, local_models

def __surgical_aggregation(K, global_model, local_models, global_labels, local_labels):
  '''Surgical aggregation for final classification Dense layer'''
  # Existing weights from global model. We'll update these!
  # A Dense layer consists of weights and bias [x' = a(W.x + B)]
  global_w, global_b = global_model.layers[-1].get_weights()
  local_weights = __get_local_weights(K, local_models, -1)
  # Surgically aggregate knowledge for each label from all clients
  for l, label in enumerate(global_labels):
    label_w = []
    label_b = []
    # Aggregate only if label exists in a client
    for k in range(K):
      if label in local_labels[k]:
        # Local index of global label
        idx = local_labels[k].index(label)
        # Store weights
        label_w += [local_weights[k][0][:,idx]]
        label_b += [local_weights[k][1][idx]]
    # Update label weights only if label was present in selected clients!
    if len(label_b) > 0:
      # Aggregated weight for label from clients containing it
      global_w[:,l] = np.mean(label_w, axis=0)
      global_b[l] = np.mean(label_b)
  # Update weights for global model
  global_model.layers[-1].set_weights([global_w, global_b])
  # Sync aggregated weights across all clients
  for k in range(K):
    client_w, client_b = local_weights[k]
    # Surgically pick only the needed weights to update local model from the global model weights
    for l, label in enumerate(local_labels[k]):
      # Map local label to global label index
      idx = global_labels.index(label)
      # Surgically pick weights
      client_w[:,l] = global_w[:,idx]
      client_b[l] = global_b[idx]
    # Sync surgically aggregated weights to local model
    local_models[k].layers[-1].set_weights([client_w, client_b])
  return global_model, local_models

# Modified implementation of keras.losses.BinaryCrossentropy()
# (https://github.com/keras-team/keras/blob/v2.13.1/keras/losses.py#L2401)
class PartialBinaryCrossentropy(keras.losses.Loss):
  def __init__(self, local_labels, global_labels):
    super().__init__()
    self.indices = tf.convert_to_tensor([global_labels.index(label) for label in local_labels])
  
  def call(
    self, 
    y_true, 
    y_pred, 
    from_logits = False, 
    label_smoothing = 0.0,
    axis = -1
  ):        
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    # Modification
    y_pred = tf.gather(y_pred, self.indices, axis=-1)
    y_true = tf.gather(y_true, self.indices, axis=-1)
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)
    def __smooth_labels():
      return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    y_true = tf.__internal__.smart_cond.smart_cond(
      label_smoothing, __smooth_labels, lambda: y_true
    )
    return backend.mean(
      backend.binary_crossentropy(y_true, y_pred, from_logits = from_logits),
      axis = axis
    )
    
def __train_federated(
  K,
  train_ds,
  val_ds,
  ckpt_dir, 
  strategy = Strategies.FedBNplus,
  surgical_aggregation = True,
  partial_loss = False,
  naive = False,
  learning_rate = 1e-5,
  epochs = 1000,
  epochs_per_step = 1,
  warmup_epochs = 15,
  image_shape = (224,224,3),
  early_stopping = True
):
  # Sanity checks!
  if K < 2:
    raise ValueError('At least 2 client required')
  if len(train_ds) != K and len(val_ds) != K:
    raise ValueError('K training and validation datasets must be passed')
  if not isinstance(learning_rate, (list, tuple, np.ndarray)):
    learning_rate = [learning_rate]*K
  os.makedirs(os.path.join(MODEL_DIR, ckpt_dir, 'top'), exist_ok=True)
  os.makedirs(os.path.join(MODEL_DIR, ckpt_dir, 'local'), exist_ok=True)
  os.makedirs(os.path.join(LOGS_DIR, ckpt_dir), exist_ok=True)
  # Load all client datasets and labels
  local_labels = []
  train_data = []
  val_data = []
  for k in range(K):
    # Sanity check!
    if train_ds[k].labels != val_ds[k].labels:
      raise ValueError('Mismatched labels!')
    local_labels += [train_ds[k].labels]
  # Combine all local labels
  global_labels = union_labels(local_labels)
  # Create and compile global model
  global_model, _ = create_model(len(global_labels), image_shape)
  global_model.compile(
    loss = keras.losses.BinaryCrossentropy(),
    metrics = keras.metrics.AUC(curve='ROC', name='auc', multi_label=True)
  )
  # Prepare data
  for k in range(K):
    # Sync tasks across all clients for naive aggregation and partial loss
    if partial_loss or naive:
      train_ds[k] = train_ds[k].expand_labels(global_labels)
      val_ds[k] = val_ds[k].expand_labels(global_labels)
    # Load train + val datasets
    train_data += [train_ds[k].get_dataset(image_shape[:2])]
    val_data += [val_ds[k].get_dataset(image_shape[:2])]
  # Sanity check!
  if surgical_aggregation:
    print(f'{str(strategy).split(".")[-1]} - Surgical aggregation')
  elif naive:
    print(f'{str(strategy).split(".")[-1]} - FL w/ naive aggregation')
  elif partial_loss:
    print(f'{str(strategy).split(".")[-1]} - FL w/ partial loss')
  else:
    print(f'{str(strategy).split(".")[-1]} - Baseline FL')
  print(f'# of clients (K) = {K}')
  print(f'# of global labels = {len(global_labels)}')
  for k in range(K):
    print(f'  # of labels for client {k+1} = {len(local_labels[k])}, # of tasks in dataset for client {k+1} = {len(train_ds[k].labels)}')
  # Keep track of all local models
  local_models = []
  # Load all "warmed-up" local models
  if not os.path.exists(os.path.join(MODEL_DIR, ckpt_dir, 'top', 'model_client1.hdf5')):
    for k in range(K):
      # Train local model classification head
      __train_local(
        train_ds[k], 
        val_ds[k], 
        os.path.join(ckpt_dir, 'top'), 
        ckpt_name = f'model_client{k+1}.hdf5', 
        warmup_epochs = warmup_epochs, 
        image_shape = image_shape,
        only_warmup_model = True
      )
  for k in range(K):
    # Path to local model
    model_path = os.path.join(ckpt_dir, 'top', f'model_client{k+1}.hdf5')
    # Load current "best" local model to use for finetuning w/ FedAvg
    model = load_model(model_path)
    # Unfreeze model layers
    model.trainable = True
    # Freeze BatchNorm layers when using FedBN+
    if strategy == Strategies.FedBNplus:
      for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
          layer.trainable = False
    # Sanity check! If FedFBN, this should print 124
    print(f'Node {k+1} - Trainable variables: {len(model.trainable_variables)}')
    # Modify loss function if partial loss is used with FL
    if not partial_loss:
      model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate[k]),
        loss = keras.losses.BinaryCrossentropy(),
        metrics = keras.metrics.AUC(curve='ROC', name='auc', multi_label=True)
      )
    else:
      model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate[k]),
        loss = PartialBinaryCrossentropy(local_labels[k], global_labels),
        metrics = keras.metrics.AUC(curve='ROC', name='auc', multi_label=True)
      )
    local_models += [model]
  logs = []
  # Monitor metric for checkpoints
  best_val_loss = None
  # Monitor for Early Stopping
  early_stopping_cfg = {
    'wait': 0,
    'patience': 25,
    'min_delta': 1e-4,
  }
  # Calculate number of steps
  steps = epochs//epochs_per_step
  # Start training!
  for i in range(steps):
    print('Step ' + str(i+1) + '/' + str(steps))
    # Finetune local model for all selected nodes
    for k in range(K):
      history = local_models[k].fit(
        train_data[k],
        validation_data = val_data[k],
        epochs = epochs_per_step,
        use_multiprocessing = True,
        verbose = 0
      )
      history = history.history
      # Training log
      for e in range(epochs_per_step):
        # step, epoch, client idx, before/after agg, metrics...
        logs += [[i+1, (i*epochs_per_step)+e+1, k+1, 0, history['loss'][e], history['auc'][e], history['val_loss'][e], history['val_auc'][e]]]
    # Update weights for all layers except the final classification layer
    for j in range(len(global_model.layers)):
      if j < len(global_model.layers)-1 or partial_loss or naive:
        # Update weights for representation block
        # Update the task block only if naive aggregation or partial loss
        if strategy == Strategies.FedAvg:
          global_model, local_models = __fedavg(K, global_model, local_models, j)
        if strategy in (Strategies.FedBN, Strategies.FedBNplus):
          global_model, local_models = __fedbn(K, global_model, local_models, j)
      else:
        # Surgical aggregation for task block
        if surgical_aggregation:
          global_model, local_models = __surgical_aggregation(K, global_model, local_models, global_labels, local_labels)
    train_loss = []
    val_loss = []
    for k in range(K):
      local_train_loss, local_train_roc = local_models[k].evaluate(train_data[k], verbose=0)
      local_val_loss, local_val_roc = local_models[k].evaluate(val_data[k], verbose=0)
      # Training log
      logs += [[i+1, (i+1)*epochs_per_step, k+1, 1, local_train_loss, local_train_roc, local_val_loss, local_val_roc]]
      # Console log
      print(f'Node {k+1} - val_loss: {np.around(local_val_loss, 4)} - val_auroc: {np.around(local_val_roc, 4)}')
      train_loss += [local_train_loss]
      val_loss += [local_val_loss]
    # Combine train + val loss from all clients to determine overall performance of global model
    train_loss = np.mean(train_loss)
    val_loss = np.mean(val_loss)
    logs += [[i+1, (i+1)*epochs_per_step, np.nan, np.nan, train_loss, np.nan, val_loss, np.nan]]
    # Only save "best" models
    if best_val_loss == None or val_loss < best_val_loss:
      print(f'Global Model - combined_val_loss: {np.around(val_loss, 4)} - New Best Validation Loss')
      if early_stopping:
        early_stopping_cfg['wait'] = 0
      best_val_loss = val_loss
      if surgical_aggregation or partial_loss or naive:
        global_model.save(os.path.join(MODEL_DIR, ckpt_dir, 'model_global.hdf5'))
      for k in range(K):
        local_models[k].save(os.path.join(MODEL_DIR, ckpt_dir, f'local/model_node{k+1}.hdf5'))
    else:
      print(f'Global Model - combined_val_loss: {np.around(val_loss, 4)}')
    if early_stopping and val_loss >= best_val_loss + early_stopping_cfg['min_delta']:
      early_stopping_cfg['wait'] += 1
      if early_stopping_cfg['wait'] > early_stopping_cfg['patience']:
        print(f'No improvement in {early_stopping_cfg["patience"]} epochs. Early stopping at step {i+1}/{steps}')
        break
  logs = pd.DataFrame(np.array(logs), columns=['step', 'epoch', 'client_idx', 'before_after_agg', 'loss', 'auc', 'val_loss', 'val_auc']).sort_values(['step', 'epoch', 'client_idx', 'before_after_agg']).to_csv(os.path.join(LOGS_DIR, ckpt_dir, f'model_logs.csv'), index=False)
   
def train_federated_baseline(
  K,
  train_ds,
  val_ds,
  ckpt_dir, 
  strategy = Strategies.FedBNplus,
  learning_rate = 1e-5,
  warmup_epochs = 15,
  epochs = 1000,
  epochs_per_step = 1,
  image_shape = (224,224,3),
  early_stopping = True
):
  return __train_federated(
    K,
    train_ds,
    val_ds,
    ckpt_dir, 
    strategy = strategy,
    surgical_aggregation = False,
    partial_loss = False,
    naive = False,
    learning_rate = learning_rate,
    warmup_epochs = warmup_epochs,
    epochs = epochs,
    epochs_per_step = epochs_per_step,
    image_shape = image_shape,
    early_stopping = early_stopping
  )
      
def train_federated_naive(
  K,
  train_ds,
  val_ds,
  ckpt_dir, 
  strategy = Strategies.FedBNplus,
  learning_rate = 1e-5,
  warmup_epochs = 15,
  epochs = 1000,
  epochs_per_step = 1,
  image_shape = (224,224,3),
  early_stopping = True
):
  return __train_federated(
    K,
    train_ds,
    val_ds,
    ckpt_dir, 
    strategy = strategy,
    surgical_aggregation = False,
    partial_loss = False,
    naive = True,
    learning_rate = learning_rate,
    warmup_epochs = warmup_epochs,
    epochs = epochs,
    epochs_per_step = epochs_per_step,
    image_shape = image_shape,
    early_stopping = early_stopping
  )
  
def train_federated_partial_loss(
  K,
  train_ds,
  val_ds,
  ckpt_dir, 
  strategy = Strategies.FedBNplus,
  learning_rate = 1e-5,
  warmup_epochs = 15,
  epochs = 1000,
  epochs_per_step = 1,
  image_shape = (224,224,3),
  early_stopping = True
):
  return __train_federated(
    K,
    train_ds,
    val_ds,
    ckpt_dir, 
    strategy = strategy,
    surgical_aggregation = False,
    partial_loss = True,
    naive = False,
    learning_rate = learning_rate,
    warmup_epochs = warmup_epochs,
    epochs = epochs,
    epochs_per_step = epochs_per_step,
    image_shape = image_shape,
    early_stopping = early_stopping
  )
  
def train_surgical_aggregation(
  K,
  train_ds,
  val_ds,
  ckpt_dir, 
  strategy = Strategies.FedBNplus,
  learning_rate = 1e-5,
  warmup_epochs = 15,
  epochs = 1000,
  epochs_per_step = 1,
  image_shape = (224,224,3),
  early_stopping = True
):
  return __train_federated(
    K,
    train_ds,
    val_ds,
    ckpt_dir, 
    strategy = strategy,
    surgical_aggregation = True,
    partial_loss = False,
    naive = False,
    learning_rate = learning_rate,
    warmup_epochs = warmup_epochs,
    epochs = epochs,
    epochs_per_step = epochs_per_step,
    image_shape = image_shape,
    early_stopping = early_stopping
  )
import os
import pandas as pd
import json

import local, federated 
from dataset import Dataset

def train_baseline():
  with open('configs/baseline_config.json', 'r') as f:
    config = json.load(f)
  for ds in config:
    # Load config for dataset
    dataset = ds['dataset']
    labels = ds['labels']
    # Checkpoint directory
    ckpt_dir = f'baseline_{dataset.lower()}/'
    # Set up train + val datasets
    splits_dir = f'splits/data/{dataset}/'
    train_ds = Dataset(
      pd.read_csv(os.path.join(splits_dir, 'train.csv')), 
      labels
    )
    val_ds = Dataset(
      pd.read_csv(os.path.join(splits_dir, 'val.csv')), 
      labels
    )
    # Train baseline model
    local.train_baseline(
      train_ds, 
      val_ds,
      ckpt_dir
    )
    
def train_simulation_experiments(exps = ['exp1.1', 'exp1.2']):
  for exp in exps:
    with open(f'configs/exp1/{exp}_config.json', 'r') as f:
      config = json.load(f)
    # Load config for dataset
    for i, sub_exp in enumerate(config['experiments']):
      # Load config for experiment
      K = sub_exp['num_nodes']
      local_labels = sub_exp['local_labels']
      # Checkpoint directory
      ckpt_dir = f'exp1/{exp}/{exp}.{i+1}/'
      def freshen_data():
        # Set up train + val datasets
        train_ds = []
        val_ds = []
        for k in range(K):
          splits_dir = os.path.join('splits/experiments', ckpt_dir, f'client_{k+1}/')
          train_ds += [
            Dataset(pd.read_csv(os.path.join(splits_dir, 'train.csv')), local_labels[k])
          ]
          val_ds += [
            Dataset(pd.read_csv(os.path.join(splits_dir, 'val.csv')), local_labels[k])
          ]
        return train_ds, val_ds
      
      # Train local central aggregation model
      train_ds, val_ds = freshen_data()
      local.train_centralized(
        train_ds, 
        val_ds,
        os.path.join(ckpt_dir, 'central_aggregation/'),
      )
      
      # Train federated naive model
      train_ds, val_ds = freshen_data()
      federated.train_federated_naive(
        K, 
        train_ds,
        val_ds,
        os.path.join(ckpt_dir, f'federated_naive_fedbnplus/'),
        strategy = federated.Strategies.FedBNplus,
        learning_rate = 5e-5
      )
      
      # Train federated partial loss model
      train_ds, val_ds = freshen_data()
      federated.train_federated_partial_loss(
        K, 
        train_ds,
        val_ds,
        os.path.join(ckpt_dir, f'federated_partial_loss_fedbnplus/'),
        strategy = federated.Strategies.FedBNplus,
        learning_rate = 5e-5
      )
      
      # Train surgical aggregation model
      train_ds, val_ds = freshen_data()
      federated.train_surgical_aggregation(
        K, 
        train_ds,
        val_ds,
        os.path.join(ckpt_dir, f'surgical_aggregation_fedbnplus/'),
        strategy = federated.Strategies.FedBNplus,
        learning_rate = 5e-5
      )
      
def train_realworld_experiments():
  with open(f'configs/exp2/config.json', 'r') as f:
    config = json.load(f)
  # Load config for dataset
  for i, exp in enumerate(config['experiments']):
    if i < 3:
      continue
    # Load config for experiment
    K = exp['num_nodes']
    local_labels = exp['local_labels']
    if isinstance(exp['dataset'], list):    
      learning_rate = [1e-5, 5e-5]
    else:
      learning_rate = [5e-5]*K   
    # Checkpoint directory
    ckpt_dir = f'exp2/exp2.{i+1}/'
    def freshen_data():
      # Set up train + val datasets
      train_ds = []
      val_ds = []
      for k in range(K):
        splits_dir = os.path.join('splits/experiments', ckpt_dir, f'client_{k+1}/')
        train_ds += [
          Dataset(pd.read_csv(os.path.join(splits_dir, 'train.csv')), local_labels[k])
        ]
        val_ds += [
          Dataset(pd.read_csv(os.path.join(splits_dir, 'val.csv')), local_labels[k])
        ]
      return train_ds, val_ds
    
    # Train local central aggregation model
    train_ds, val_ds = freshen_data()
    local.train_centralized(
      train_ds, 
      val_ds,
      os.path.join(ckpt_dir, 'central_aggregation/'),
    )
    
    for strategy in federated.Strategies:
      strategy_key = str(strategy).split('.')[-1].lower()
      # Train federated baseline model
      train_ds, val_ds = freshen_data()
      federated.train_federated_baseline(
        K, 
        train_ds,
        val_ds,
        os.path.join(ckpt_dir, f'federated_baseline_{strategy_key}/'),
        strategy = strategy,
        learning_rate = learning_rate
      )
      
      # FedBN is for personalized FL only. No need to train "global" 
      if strategy != federated.Strategies.FedBN:
        # Train federated naive model
        train_ds, val_ds = freshen_data()
        federated.train_federated_naive(
          K, 
          train_ds,
          val_ds,
          os.path.join(ckpt_dir, f'federated_naive_{strategy_key}/'),
          strategy = strategy,
          learning_rate = learning_rate
        )
        
        # Train federated partial loss model
        train_ds, val_ds = freshen_data()
        federated.train_federated_partial_loss(
          K, 
          train_ds,
          val_ds,
          os.path.join(ckpt_dir, f'federated_partial_loss_{strategy_key}/'),
          strategy = strategy,
          learning_rate = learning_rate
        )
        
        if strategy == federated.Strategies.FedAvg:
          continue
        # Train surgical aggregation model
        train_ds, val_ds = freshen_data()
        federated.train_surgical_aggregation(
          K, 
          train_ds,
          val_ds,
          os.path.join(ckpt_dir, f'surgical_aggregation_{strategy_key}_NEW/'),
          strategy = strategy,
          learning_rate = learning_rate,
          epochs = 150,
          early_stopping = False
        )
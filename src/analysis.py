import os
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm.auto import tqdm
import json
from functools import partial
import multiprocessing

from federated import Strategies
from dataset import union_labels

__ALL_LABELS = [
  'Atelectasis',
  'Cardiomegaly',
  'Consolidation',
  'Edema',
  'Emphysema',
  'Enlarged Cardiomediastinum',
  'Fibrosis',
  'Fracture',
  'Hernia',
  'Infiltration',
  'Lung Lesion',
  'Lung Opacity',
  'Mass',
  'Nodule',
  'Pleural Effusion',
  'Pleural Other',
  'Pleural Thickening',
  'Pneumonia',
  'Pneumothorax',
  'Support Devices'
]

# Bootstrapped Metrics

def __bce_loss(y_true, y_pred):
  alpha = (1-y_true) * np.log(1-y_pred)
  beta = y_true * np.log(y_pred)
  return -np.mean(alpha + beta)

def __auroc(y_true, y_pred):
  return metrics.roc_auc_score(y_true, y_pred)

def __label_auroc(y_true, y_pred):
  return metrics.roc_auc_score(y_true, y_pred, average=None)

def __bootstrapped_metrics(y_true, y_pred, idx):
  loss = __bce_loss(y_true[idx], y_pred[idx])
  auc = __auroc(y_true[idx], y_pred[idx])
  return loss, auc

def bootstrapped_metrics(y_true, y_pred, p=0.05, n_iterations=100, seed=1337, raw_scores=False):
  if y_true.shape != y_pred.shape:
    raise ValueError('Whoops!')
  # Test set size
  n_samples = len(y_true)
  idx = np.random.RandomState(seed).choice(np.arange(n_samples), (n_iterations, n_samples), replace=True)
  # Use multithreading to speed things up
  count = multiprocessing.cpu_count()
  with multiprocessing.Pool(processes=count) as pool:
    scores = np.array(pool.map(partial(__bootstrapped_metrics, y_true, y_pred), idx))
  # scores = np.sort(scores[scores != None])
  if raw_scores:
    # For debugging
    return scores
  else:
    mean = np.nanmean(scores, axis=0)
    ll, ul = np.nanquantile(scores, q=(p/2, 1-p/2), axis=0)
    return mean, (ll, ul)

def analyze_baseline():
  with open('configs/dataset_config.json', 'r') as f:
    label_config = json.load(f)
  with open('configs/baseline_config.json', 'r') as f:
    config = json.load(f)
  summary = []
  for ds in tqdm(config):
    # Load config for dataset
    train_dataset = ds['dataset']
    train_labels = ds['labels']
    model_type = f'baseline_{train_dataset.lower()}'
    for test_dataset in ['NIH', 'CheXpert', 'MIMIC']:
      test_labels = label_config[test_dataset]['labels']
      labels = np.intersect1d(train_labels, test_labels)
      # Ground truth
      y_true = pd.read_csv(f'splits/data/{test_dataset}/test.csv')[labels].values
      # Predictions
      y_pred = pd.read_csv(f'results/raw/{model_type}/{test_dataset}_pred.csv')[labels].values
      # Metrics
      label_auc = np.zeros(len(__ALL_LABELS))
      label_auc.fill(np.nan)
      label_auc[np.in1d(__ALL_LABELS, labels)] = __label_auroc(y_true, y_pred)
      mean_auc = label_auc[np.in1d(__ALL_LABELS, labels)].mean()
      sd_auc = label_auc[np.in1d(__ALL_LABELS, labels)].std()
      summary += [[model_type, '', test_dataset, np.nan, mean_auc, sd_auc] + label_auc.tolist()]
  pd.DataFrame(np.array(summary, dtype=object), columns=['model_type', 'strategy', 'test_dataset', 'node', 'mean_auc', 'sd_auc'] + __ALL_LABELS).to_csv(f'results/summary/baseline.csv', index=False)
      
def analyze_simulation_experiments():
  with open('configs/dataset_config.json', 'r') as f:
    label_config = json.load(f)
  for exp in ['exp1.1', 'exp1.2']:
    with open(f'configs/exp1/{exp}_config.json', 'r') as f:
      config = json.load(f)
    # Load config for dataset
    for i, sub_exp in enumerate(tqdm(config['experiments'])):
      # Load config for experiment
      train_labels = union_labels(sub_exp['local_labels'])
      # Experiment directory
      exp_dir = f'exp1/{exp}/'
      os.makedirs(f'results/summary/{exp_dir}/', exist_ok=True)
      summary = []
      for test_dataset in ['NIH', 'MIMIC']:
        test_labels = label_config[test_dataset]['labels']
        labels = np.intersect1d(train_labels, test_labels)
        # Ground truth
        y_true = pd.read_csv(f'splits/data/{test_dataset}/test.csv')[labels].values
        
        # Test local central aggregation model
        model_type = 'central_aggregation'
        y_pred = pd.read_csv(f'results/raw/{exp_dir}/{exp}.{i+1}/{model_type}/{test_dataset}_pred.csv')[labels].values
        label_auc = np.zeros(len(__ALL_LABELS))
        label_auc.fill(np.nan)
        label_auc[np.in1d(__ALL_LABELS, labels)] = __label_auroc(y_true, y_pred)
        mean_auc = label_auc[np.in1d(__ALL_LABELS, labels)].mean()
        sd_auc = label_auc[np.in1d(__ALL_LABELS, labels)].std()
        summary += [[model_type, '', test_dataset, np.nan, mean_auc, sd_auc] + label_auc.tolist()]
        
        # Test federated naive model
        model_type = 'federated_naive'
        y_pred = pd.read_csv(f'results/raw/{exp_dir}/{exp}.{i+1}/{model_type}_fedbnplus/{test_dataset}_pred.csv')[labels].values
        label_auc = np.zeros(len(__ALL_LABELS))
        label_auc.fill(np.nan)
        label_auc[np.in1d(__ALL_LABELS, labels)] = __label_auroc(y_true, y_pred)
        mean_auc = label_auc[np.in1d(__ALL_LABELS, labels)].mean()
        sd_auc = label_auc[np.in1d(__ALL_LABELS, labels)].std()
        summary += [[model_type, 'fedbnplus', test_dataset, np.nan, mean_auc, sd_auc] + label_auc.tolist()]
        
        # Test federated partial loss model
        model_type = 'federated_partial_loss'
        y_pred = pd.read_csv(f'results/raw/{exp_dir}/{exp}.{i+1}/{model_type}_fedbnplus/{test_dataset}_pred.csv')[labels].values
        label_auc = np.zeros(len(__ALL_LABELS))
        label_auc.fill(np.nan)
        label_auc[np.in1d(__ALL_LABELS, labels)] = __label_auroc(y_true, y_pred)
        mean_auc = label_auc[np.in1d(__ALL_LABELS, labels)].mean()
        sd_auc = label_auc[np.in1d(__ALL_LABELS, labels)].std()
        summary += [[model_type, 'fedbnplus', test_dataset, np.nan, mean_auc, sd_auc] + label_auc.tolist()]
        
        # Test surgical aggregation model
        model_type = 'surgical_aggregation'
        y_pred = pd.read_csv(f'results/raw/{exp_dir}/{exp}.{i+1}/{model_type}_fedbnplus/{test_dataset}_pred.csv')[labels].values
        label_auc = np.zeros(len(__ALL_LABELS))
        label_auc.fill(np.nan)
        label_auc[np.in1d(__ALL_LABELS, labels)] = __label_auroc(y_true, y_pred)
        mean_auc = label_auc[np.in1d(__ALL_LABELS, labels)].mean()
        sd_auc = label_auc[np.in1d(__ALL_LABELS, labels)].std()
        summary += [[model_type, 'fedbnplus', test_dataset, np.nan, mean_auc, sd_auc] + label_auc.tolist()]
      pd.DataFrame(np.array(summary, dtype=object), columns=['model_type', 'strategy', 'test_dataset', 'node', 'mean_auc', 'sd_auc'] + __ALL_LABELS).to_csv(f'results/summary/{exp_dir}/{exp}.{i+1}.csv', index=False)
      
def analyze_realworld_experiments():
  with open('configs/dataset_config.json', 'r') as f:
    label_config = json.load(f)
  with open(f'configs/exp2/config.json', 'r') as f:
    config = json.load(f)
  # Load config for dataset
  for i, exp in enumerate(tqdm(config['experiments'])):
    # Load config for experiment
    K = exp['num_nodes']
    local_labels = exp['local_labels']
    train_labels = union_labels(local_labels)
    # Experiment directory
    exp_dir = f'exp2/exp2.{i+1}'
    os.makedirs(f'results/summary/exp2/', exist_ok=True)
    summary = []
    for test_dataset in ['NIH', 'CheXpert', 'MIMIC']:
      if test_dataset == 'CheXpert' and i < 2:
        continue
      test_labels = label_config[test_dataset]['labels']
      labels = np.intersect1d(train_labels, test_labels)
      # Ground truth
      y_true = pd.read_csv(f'splits/data/{test_dataset}/test.csv')[labels].values
        
      # Test baseline models
      model_type = 'baseline' 
      for ds in ['NIH', 'CheXpert']:
        if ds == 'CheXpert' and i < 2:
          continue
        if i == 3:
          continue
        y_pred = pd.read_csv(f'results/raw/{model_type}_{ds.lower()}/{test_dataset}_pred.csv')[labels].values
        label_auc = np.zeros(len(__ALL_LABELS))
        label_auc.fill(np.nan)
        label_auc[np.in1d(__ALL_LABELS, labels)] = __label_auroc(y_true, y_pred)
        mean_auc = label_auc[np.in1d(__ALL_LABELS, labels)].mean()
        sd_auc = label_auc[np.in1d(__ALL_LABELS, labels)].std()
        summary += [[f'{model_type}_{ds.lower()}', '', test_dataset, np.nan, mean_auc, sd_auc] + label_auc.tolist()]
        
      # Test local central aggregation model
      model_type = 'central_aggregation'
      y_pred = pd.read_csv(f'results/raw/{exp_dir}/{model_type}/{test_dataset}_pred.csv')[labels].values
      label_auc = np.zeros(len(__ALL_LABELS))
      label_auc.fill(np.nan)
      label_auc[np.in1d(__ALL_LABELS, labels)] = __label_auroc(y_true, y_pred)
      mean_auc = label_auc[np.in1d(__ALL_LABELS, labels)].mean()
      sd_auc = label_auc[np.in1d(__ALL_LABELS, labels)].std()
      summary += [[model_type, '', test_dataset, np.nan, mean_auc, sd_auc] + label_auc.tolist()]
      
      # Federated models
      for strategy in Strategies:
        strategy_key = str(strategy).split('.')[-1].lower()
        
        # Test federated baseline model
        model_type = f'federated_baseline'
        for k in range(K):
          labels_t = np.intersect1d(local_labels[k], test_labels)
          y_true_t = pd.read_csv(f'splits/data/{test_dataset}/test.csv')[labels_t].values
          y_pred = pd.read_csv(f'results/raw/{exp_dir}/{model_type}_{strategy_key}/{test_dataset}_node{k+1}_pred.csv')[labels_t].values
          label_auc = np.zeros(len(__ALL_LABELS))
          label_auc.fill(np.nan)
          label_auc[np.in1d(__ALL_LABELS, labels_t)] = __label_auroc(y_true_t, y_pred)
          mean_auc = label_auc[np.in1d(__ALL_LABELS, labels_t)].mean()
          sd_auc = label_auc[np.in1d(__ALL_LABELS, labels_t)].std()
          summary += [[model_type, strategy_key, test_dataset, k+1, mean_auc, sd_auc] + label_auc.tolist()]
          
        # FedBN is for personalized FL only. No need to test "global" 
        if strategy != Strategies.FedBN:
          
          # Test federated naive model
          model_type = f'federated_naive'
          y_pred = pd.read_csv(f'results/raw/{exp_dir}/{model_type}_{strategy_key}/{test_dataset}_pred.csv')[labels].values
          label_auc = np.zeros(len(__ALL_LABELS))
          label_auc.fill(np.nan)
          label_auc[np.in1d(__ALL_LABELS, labels)] = __label_auroc(y_true, y_pred)
          mean_auc = label_auc[np.in1d(__ALL_LABELS, labels)].mean()
          sd_auc = label_auc[np.in1d(__ALL_LABELS, labels)].std()
          summary += [[model_type, strategy_key, test_dataset, np.nan, mean_auc, sd_auc] + label_auc.tolist()]
          
          # Test federated partial loss model
          
          model_type = f'federated_partial_loss'
          y_pred = pd.read_csv(f'results/raw/{exp_dir}/{model_type}_{strategy_key}/{test_dataset}_pred.csv')[labels].values
          label_auc = np.zeros(len(__ALL_LABELS))
          label_auc.fill(np.nan)
          label_auc[np.in1d(__ALL_LABELS, labels)] = __label_auroc(y_true, y_pred)
          mean_auc = label_auc[np.in1d(__ALL_LABELS, labels)].mean()
          sd_auc = label_auc[np.in1d(__ALL_LABELS, labels)].std()
          summary += [[model_type, strategy_key, test_dataset, np.nan, mean_auc, sd_auc] + label_auc.tolist()]
          
          # Test surgical aggregation model
          model_type = f'surgical_aggregation'
          y_pred = pd.read_csv(f'results/raw/{exp_dir}/{model_type}_{strategy_key}/{test_dataset}_pred.csv')[labels].values
          label_auc = np.zeros(len(__ALL_LABELS))
          label_auc.fill(np.nan)
          label_auc[np.in1d(__ALL_LABELS, labels)] = __label_auroc(y_true, y_pred)
          mean_auc = label_auc[np.in1d(__ALL_LABELS, labels)].mean()
          sd_auc = label_auc[np.in1d(__ALL_LABELS, labels)].std()
          summary += [[model_type, strategy_key, test_dataset, np.nan, mean_auc, sd_auc] + label_auc.tolist()]
    pd.DataFrame(np.array(summary, dtype=object), columns=['model_type', 'strategy', 'test_dataset', 'node', 'mean_auc', 'sd_auc'] + __ALL_LABELS).to_csv(f'results/summary/{exp_dir}.csv', index=False)
          
def analyze_models():
  os.makedirs(f'results/summary/', exist_ok=True)
  analyze_baseline()
  analyze_simulation_experiments()
  analyze_realworld_experiments()
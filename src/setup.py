'''
Initialize all data for training
'''

import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import json

SEED = 1337

# Create path to each image using subject id, study id, and dicom id
def join_mimic_path(arg, format='jpg'):
  dicom_id, subject_id, study_id = arg
  return f'files/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.{format}'

def generate_labels(ds_config, key):
  ds = ds_config[key]
  raw_df = pd.read_csv(ds['path_to_raw_labels'])
  os.makedirs(f'splits/data/{key}/', exist_ok=True)
  if key == 'NIH':
    raw_df = raw_df.values
    df = []
    # Since NIH gives findings as string, we have to convert to 0 and 1s
    for j in range(raw_df.shape[0]):
      # Split by separator and assign 1 to each label found in the raw df
      # All others stay 0
      raw_labels = raw_df[j, 1].split('|')
      labels = np.zeros(14, dtype=int)
      for raw_label in raw_labels:
        # Harmonize label names
        if raw_label == 'Effusion':
          raw_label = 'Pleural Effusion'
        elif raw_label == 'Pleural_Thickening':
          raw_label = 'Pleural Thickening'
        if raw_label == 'No Finding':
          continue
        labels[ds['labels'].index(raw_label)] = 1
        # Very inefficient since this has to be processed line-by-line :/
      df += [raw_df[j, [0,3]].tolist() + labels.tolist()]
    df = pd.DataFrame(np.array(df), columns=['path', 'patient_id'] + ds['labels'])
    df[ds['labels']] = df[ds['labels']].astype(int)
  elif key == 'CheXpert':
    # Filter out all lateral images
    raw_df = raw_df[raw_df['Frontal/Lateral'] == 'Frontal']
    # Create image path and patient id from original raw path
    split_paths = raw_df['Path'].str.split('/', expand=True)
    paths = split_paths[[2,3,4]].agg('/'.join, axis=1)
    paths.name = 'path'
    patient_ids = split_paths[2].str.replace('patient', '').astype(int)
    patient_ids.name = 'patient_id'
    # Convert labels to int, fill all missing values with 0
    # Replace all uncertain labels with 0 (U-Zeros approach)
    labels = raw_df[ds['labels']].fillna(0).replace(-1, 0).astype(int)
    # Concatenate all columns and save
    df = pd.concat((paths, patient_ids, labels), axis=1)
  elif key == 'MIMIC':
    # MIMIC provides labels and metadata is two diff csv files, merge on keys
    raw_df = pd.merge(raw_df, pd.read_csv(ds['path_to_raw_metadata']), on=['subject_id', 'study_id'])
    # Convert labels to int, fill all missing values with 0
    # Replace all uncertain labels with 0 (U-Zeros approach)
    raw_df[ds['labels']] = raw_df[ds['labels']].fillna(0).replace(-1, 0).astype(int)
    # Filter out all lateral images
    raw_df = raw_df[raw_df['ViewPosition'].isin(['AP', 'PA'])]
    raw_df['path'] = raw_df[['dicom_id', 'subject_id', 'study_id']].apply(join_mimic_path, axis=1)
    raw_df['patient_id'] = raw_df['subject_id']
    # Filter out all other columns
    df = pd.DataFrame(raw_df[['path', 'patient_id'] + ds['labels']])
  # Append path to location of images to image path
  df['path'] = ds['path_to_images'] + df['path']
  # Sort and save
  df.sort_values('path').to_csv(f'splits/data/{key}/labels.csv', index=False)

def split_by_pos_neg(df):
  # Group by and identify which patients have been diagnosed with a disease across all studies
  tdf = df.groupby('patient_id').sum().reset_index()
  if (df.columns[1:] != tdf.columns).all():
    raise ValueError('Whoops! Check the abnormal calculation')
  # Quantize to 0 or 1
  tdf['Abnormal'] = (tdf[list(tdf.columns[1:])].sum(axis=1) > 0).astype(int)
  # Split by pos/neg patients
  pos_ids = tdf[tdf['Abnormal'] == 1]['patient_id'].unique()
  neg_ids = tdf[tdf['Abnormal'] == 0]['patient_id'].unique()
  return pos_ids, neg_ids

def generate_global_splits(key, splits=(0.7, 0.1, 0.2)):
  if sum(splits) != 1:
    raise ValueError('Sum must add up to 1')
  # Create splits for all datasets (except mimic, external test)
  df = pd.read_csv(f'splits/data/{key}/labels.csv')
  if key == 'MIMIC':
    df.to_csv(f'splits/data/{key}/test.csv', index=False)
  else:
    np.random.seed(SEED)
    # Equally distribute pos/neg patients across each split
    # Labels are flipped since this is the "normal" label
    # Here, pos and neg refer to presence of abnormality
    pos_ids, neg_ids = split_by_pos_neg(df)
    # Randomly shuffle patient ids
    np.random.shuffle(pos_ids)
    np.random.shuffle(neg_ids)
    # Size of each pos split
    n_pos_train, n_pos_val, _ = (np.array(splits)*len(pos_ids)).astype(int) 
    # Get pos patient ids for each split
    pos_train_val_ids = pos_ids[:n_pos_train + n_pos_val]
    pos_test_ids = pos_ids[n_pos_train + n_pos_val:]
    # Size of each neg split
    n_neg_train, n_neg_val, _ = (np.array(splits)*len(neg_ids)).astype(int) 
    # Get neg patient ids for each split
    neg_train_val_ids = neg_ids[:n_neg_train + n_neg_val]
    neg_test_ids = neg_ids[n_neg_train + n_neg_val:]
    # Concatenate both pos and neg patient ids for global test split
    test_ids = np.concatenate((pos_test_ids, neg_test_ids))
    # Save global test split
    df[df['patient_id'].isin(test_ids)].to_csv(f'splits/data/{key}/test.csv', index=False)
    # Randomly shuffle patient ids
    np.random.shuffle(pos_train_val_ids)
    np.random.shuffle(neg_train_val_ids)
    # Get pos patient ids for each split
    pos_train_ids = pos_train_val_ids[:n_pos_train]
    pos_val_ids = pos_train_val_ids[n_pos_train:]
    # Get neg patient ids for each split
    neg_train_ids = neg_train_val_ids[:n_neg_train]
    neg_val_ids = neg_train_val_ids[n_neg_train:]
    # Concatenate both pos and neg patient ids for each split
    train_ids = np.concatenate((pos_train_ids, neg_train_ids))
    val_ids = np.concatenate((pos_val_ids, neg_val_ids))
    # Save all splits
    df[df['patient_id'].isin(train_ids)].to_csv(f'splits/data/{key}/train.csv', index=False)
    df[df['patient_id'].isin(val_ids)].to_csv(f'splits/data/{key}/val.csv', index=False)

def split_for_K_nodes(K, train_df, val_df):
  # Optional arg to control dataset size
  # If N is not specified, dataset will be divided into K equal parts
  np.random.seed(SEED)
  # Split train data
  pos_train_ids, neg_train_ids = split_by_pos_neg(train_df)
  np.random.shuffle(pos_train_ids)
  np.random.shuffle(neg_train_ids)
  n_pos, n_neg = int(1/K*len(pos_train_ids)), int(1/K*len(neg_train_ids))
  node_train_ids = []
  for i in range(K):
    node_train_ids += [np.concatenate((pos_train_ids[i*n_pos:(i+1)*n_pos], neg_train_ids[i*n_neg:(i+1)*n_neg]))]
  # Split val data
  pos_val_ids, neg_val_ids = split_by_pos_neg(val_df)
  np.random.shuffle(pos_val_ids)
  np.random.shuffle(neg_val_ids)
  n_pos, n_neg = int(1/K*len(pos_val_ids)), int(1/K*len(neg_val_ids))
  node_val_ids = []
  for i in range(K):
    node_val_ids += [np.concatenate((pos_val_ids[i*n_pos:(i+1)*n_pos], neg_val_ids[i*n_neg:(i+1)*n_neg]))]
  return node_train_ids, node_val_ids

def generate_simulation_splits():
  train_df = pd.read_csv('splits/data/NIH/train.csv')
  val_df = pd.read_csv('splits/data/NIH/val.csv')
  # Experiment 1
  # Simulation Experiments -- Surgical Aggregation w/ FedBN+ vs Baseline, Central, FL w/ FedBN+, FL w/ Partial Loss & FedBN+
  os.makedirs(f'splits/experiments/exp1/', exist_ok=True)
  # Exp 1.1: Number of clients
  # Exp 1.2: Label heterogeneity
  for exp in tqdm(['exp1.1', 'exp1.2']):
    os.makedirs(f'splits/experiments/exp1/{exp}/', exist_ok=True)
    with open(f'configs/exp1/{exp}_config.json', 'r') as f:
      config = json.load(f)
    for i, sub_exp in enumerate(config['experiments']):
      os.makedirs(f'splits/experiments/exp1/{exp}/{exp}.{i+1}', exist_ok=True)
      K = sub_exp['num_nodes']
      local_labels = sub_exp['local_labels']
      if len(local_labels) != K:
        raise ValueError('Invalid config! Length of local labels must match number of clients')
      node_train_ids, node_val_ids = split_for_K_nodes(K, train_df, val_df)
      for k in range(K):
        os.makedirs(f'splits/experiments/exp1/{exp}/{exp}.{i+1}/client_{k+1}/', exist_ok=True)
        train_df[train_df['patient_id'].isin(node_train_ids[k])][['path', 'patient_id'] + local_labels[k]].to_csv(f'splits/experiments/exp1/{exp}/{exp}.{i+1}/client_{k+1}/train.csv', index=False)
        val_df[val_df['patient_id'].isin(node_val_ids[k])][['path', 'patient_id'] + local_labels[k]].to_csv(f'splits/experiments/exp1/{exp}/{exp}.{i+1}/client_{k+1}/val.csv', index=False)
    
def generate_realworld_splits():
  # Experiment 2
  # Real-world Experiments -- Surgical Aggregation vs Baseline, Central, FL, FL w/ Partial Loss (with FedAvg, FedBN [local & global], FedBN+)
  os.makedirs(f'splits/experiments/exp2/', exist_ok=True)
  with open(f'configs/exp2/config.json', 'r') as f:
    config = json.load(f)
  # Exp 2.1: iid w/ complete labels
  # Exp 2.2: iid w/ partial labels
  # Exp 2.3: non-iid w/ complete labels
  # Exp 2.4: non-iid w/ partial labels 
  for i, exp in enumerate(tqdm(config['experiments'])):
    os.makedirs(f'splits/experiments/exp2/exp2.{i+1}/', exist_ok=True)
    dataset = exp['dataset']
    K = exp['num_nodes']
    local_labels = exp['local_labels']
    if len(local_labels) != K:
      raise ValueError('Invalid config! Length of local labels must match number of clients')
    if isinstance(dataset, list):
      if len(dataset) != K:
        raise ValueError('Invalid config! Length of datasets must match number of clients')
      for k in range(K):
        os.makedirs(f'splits/experiments/exp2/exp2.{i+1}/client_{k+1}/', exist_ok=True)
        train_df = pd.read_csv(f'splits/data/{dataset[k]}/train.csv')
        val_df = pd.read_csv(f'splits/data/{dataset[k]}/val.csv')
        train_df[['path', 'patient_id'] + local_labels[k]].to_csv(f'splits/experiments/exp2/exp2.{i+1}/client_{k+1}/train.csv', index=False)
        val_df[['path', 'patient_id'] + local_labels[k]].to_csv(f'splits/experiments/exp2/exp2.{i+1}/client_{k+1}/val.csv', index=False)
    else:
      train_df = pd.read_csv(f'splits/data/{dataset}/train.csv')
      val_df = pd.read_csv(f'splits/data/{dataset}/val.csv')
      node_train_ids, node_val_ids = split_for_K_nodes(K, train_df, val_df)
      for k in range(K):
        os.makedirs(f'splits/experiments/exp2/exp2.{i+1}/client_{k+1}/', exist_ok=True)
        train_df[train_df['patient_id'].isin(node_train_ids[k])][['path', 'patient_id'] + local_labels[k]].to_csv(f'splits/experiments/exp2/exp2.{i+1}/client_{k+1}/train.csv', index=False)
        val_df[val_df['patient_id'].isin(node_val_ids[k])][['path', 'patient_id'] + local_labels[k]].to_csv(f'splits/experiments/exp2/exp2.{i+1}/client_{k+1}/val.csv', index=False)
  
def setup_data(data_splits=(0.7, 0.1, 0.2)):
  with open('configs/dataset_config.json', 'r') as f:
    ds_config = json.load(f)
  print('Generating metadata')
  for key in tqdm(ds_config.keys()):
    generate_labels(ds_config, key)
    generate_global_splits(key, splits=data_splits)
  print('Generating splits for simulation experiments')
  generate_simulation_splits()
  print('Generating splits for real-world experiments')
  generate_realworld_splits()
  
if __name__=='__main__':
  os.chdir('/'.join(os.path.abspath(__file__).split('/')[:-2]))
  setup_data()
import os
import pandas as pd
from tqdm.auto import tqdm
import json

import federated, utils
from dataset import Dataset, union_labels

def test_baseline():
  with open('configs/dataset_config.json', 'r') as f:
    label_config = json.load(f)
  with open('configs/baseline_config.json', 'r') as f:
    config = json.load(f)
  for ds in tqdm(config):
    # Load config for dataset
    train_dataset = ds['dataset']
    labels = ds['labels']
    model_type = f'baseline_{train_dataset.lower()}'
    os.makedirs(f'results/raw/{model_type}/', exist_ok=True)
    for test_dataset in ['NIH', 'CheXpert', 'MIMIC']:
      # Set up test data
      test_ds = Dataset(
        pd.read_csv(f'splits/data/{test_dataset}/test.csv'), 
        label_config[test_dataset]['labels']
      )
      # Load model
      model = utils.load_model(f'{model_type}/model.hdf5')
      # Test baseline model
      y_pred = model.predict(test_ds.get_dataset(shuffle=False))
      # Save predictions
      df = pd.DataFrame(test_ds.df['path'])
      df[labels] = y_pred
      df.to_csv(f'results/raw/{model_type}/{test_dataset}_pred.csv', index=False)
    
def test_simulation_experiments():
  with open('configs/dataset_config.json', 'r') as f:
    label_config = json.load(f)
  for exp in ['exp1.1', 'exp1.2']:
    with open(f'configs/exp1/{exp}_config.json', 'r') as f:
      config = json.load(f)
    # Load config for dataset
    for i, sub_exp in enumerate(tqdm(config['experiments'])):
      # Load config for experiment
      labels = union_labels(sub_exp['local_labels'])
      # Experiment directory
      exp_dir = f'exp1/{exp}/{exp}.{i+1}'
      os.makedirs(f'results/raw/{exp_dir}/', exist_ok=True)
      for test_dataset in ['NIH', 'MIMIC']:
        # Set up test data
        test_ds = Dataset(
          pd.read_csv(f'splits/data/{test_dataset}/test.csv'), 
          label_config[test_dataset]['labels']
        )
        
        # Test local central aggregation model
        model_type = 'central_aggregation'
        os.makedirs(f'results/raw/{exp_dir}/{model_type}/', exist_ok=True)
        model = utils.load_model(f'{exp_dir}/{model_type}/model.hdf5')
        y_pred = model.predict(test_ds.get_dataset(shuffle=False))
        df = pd.DataFrame(test_ds.df['path'])
        df[labels] = y_pred
        df.to_csv(f'results/raw/{exp_dir}/{model_type}/{test_dataset}_pred.csv', index=False)
        
        # Test federated naive model
        model_type = 'federated_naive_fedbnplus'
        os.makedirs(f'results/raw/{exp_dir}/{model_type}/', exist_ok=True)
        model = utils.load_model(f'{exp_dir}/{model_type}/model_global.hdf5')
        y_pred = model.predict(test_ds.get_dataset(shuffle=False))
        df = pd.DataFrame(test_ds.df['path'])
        df[labels] = y_pred
        df.to_csv(f'results/raw/{exp_dir}/{model_type}/{test_dataset}_pred.csv', index=False)
        
        # Test federated partial loss model
        model_type = 'federated_partial_loss_fedbnplus'
        os.makedirs(f'results/raw/{exp_dir}/{model_type}/', exist_ok=True)
        model = utils.load_model(f'{exp_dir}/{model_type}/model_global.hdf5')
        y_pred = model.predict(test_ds.get_dataset(shuffle=False))
        df = pd.DataFrame(test_ds.df['path'])
        df[labels] = y_pred
        df.to_csv(f'results/raw/{exp_dir}/{model_type}/{test_dataset}_pred.csv', index=False)
        
        # Test surgical aggregation model
        model_type = 'surgical_aggregation_fedbnplus'
        os.makedirs(f'results/raw/{exp_dir}/{model_type}/', exist_ok=True)
        model = utils.load_model(f'{exp_dir}/{model_type}/model_global.hdf5')
        y_pred = model.predict(test_ds.get_dataset(shuffle=False))
        df = pd.DataFrame(test_ds.df['path'])
        df[labels] = y_pred
        df.to_csv(f'results/raw/{exp_dir}/{model_type}/{test_dataset}_pred.csv', index=False)
      
def test_realworld_experiments():
  with open('configs/dataset_config.json', 'r') as f:
    label_config = json.load(f)
  with open(f'configs/exp2/config.json', 'r') as f:
    config = json.load(f)
  # Load config for dataset
  for i, exp in enumerate(tqdm(config['experiments'])):
    # Load config for experiment
    K = exp['num_nodes']
    local_labels = exp['local_labels']
    global_labels = union_labels(local_labels)
    # Experiment directory
    exp_dir = f'exp2/exp2.{i+1}'
    os.makedirs(f'results/raw/{exp_dir}/', exist_ok=True)
    for test_dataset in ['NIH', 'CheXpert', 'MIMIC']:
      if test_dataset == 'CheXpert' and i < 2:
        continue
      # Set up train data
      test_ds = Dataset(
        pd.read_csv(f'splits/data/{test_dataset}/test.csv'), 
        label_config[test_dataset]['labels']
      )
      
      # Test local central aggregation model
      model_type = 'central_aggregation'
      os.makedirs(f'results/raw/{exp_dir}/{model_type}/', exist_ok=True)
      model = utils.load_model(f'{exp_dir}/{model_type}/model.hdf5')
      y_pred = model.predict(test_ds.get_dataset(shuffle=False))
      df = pd.DataFrame(test_ds.df['path'])
      df[global_labels] = y_pred
      df.to_csv(f'results/raw/{exp_dir}/{model_type}/{test_dataset}_pred.csv', index=False)
      
      # Federated models
      for strategy in federated.Strategies:
        strategy_key = str(strategy).split('.')[-1].lower()
        
        # Test federated baseline model
        model_type = f'federated_baseline_{strategy_key}'
        os.makedirs(f'results/raw/{exp_dir}/{model_type}/', exist_ok=True)
        for k in range(K):
          model = utils.load_model(f'{exp_dir}/{model_type}/local/model_node{k+1}.hdf5')
          y_pred = model.predict(test_ds.get_dataset(shuffle=False))
          df = pd.DataFrame(test_ds.df['path'])
          df[local_labels[k]] = y_pred
          df.to_csv(f'results/raw/{exp_dir}/{model_type}/{test_dataset}_node{k+1}_pred.csv', index=False)
          
        # FedBN is for personalized FL only. No need to test "global" 
        if strategy != federated.Strategies.FedBN:
          # Test federated naive model
          model_type = f'federated_naive_{strategy_key}'
          os.makedirs(f'results/raw/{exp_dir}/{model_type}/', exist_ok=True)
          model = utils.load_model(f'{exp_dir}/{model_type}/model_global.hdf5')
          y_pred = model.predict(test_ds.get_dataset(shuffle=False))
          df = pd.DataFrame(test_ds.df['path'])
          df[global_labels] = y_pred
          df.to_csv(f'results/raw/{exp_dir}/{model_type}/{test_dataset}_pred.csv', index=False)
          
          # Test federated partial loss model
          model_type = f'federated_partial_loss_{strategy_key}'
          os.makedirs(f'results/raw/{exp_dir}/{model_type}/', exist_ok=True)
          model = utils.load_model(f'{exp_dir}/{model_type}/model_global.hdf5')
          y_pred = model.predict(test_ds.get_dataset(shuffle=False))
          df = pd.DataFrame(test_ds.df['path'])
          df[global_labels] = y_pred
          df.to_csv(f'results/raw/{exp_dir}/{model_type}/{test_dataset}_pred.csv', index=False)
          
          # Test surgical aggregation model
          model_type = f'surgical_aggregation_{strategy_key}'
          os.makedirs(f'results/raw/{exp_dir}/{model_type}/', exist_ok=True)
          model = utils.load_model(f'{exp_dir}/{model_type}/model_global.hdf5')
          y_pred = model.predict(test_ds.get_dataset(shuffle=False))
          df = pd.DataFrame(test_ds.df['path'])
          df[global_labels] = y_pred
          df.to_csv(f'results/raw/{exp_dir}/{model_type}/{test_dataset}_pred.csv', index=False)
          
def test_models():
  test_baseline()
  test_simulation_experiments()
  test_realworld_experiments()
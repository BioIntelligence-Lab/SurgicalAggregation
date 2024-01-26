'''
Configures and returns a tf.Dataset
'''

import numpy as np
import pandas as pd
from functools import reduce
import imflow

def create_dataset(df, X, y, image_shape=(224,224), seed=1337, batch_size=64, buffer_size=32, shuffle=True):
  ds = imflow.image_dataset_from_dataframe(
    df, X, y,
    label_mode = 'multi_label',
    image_size = image_shape,
    batch_size = batch_size,
    seed = seed,
    color_mode = 'rgb',
    resize_with_pad = True,
    shuffle = shuffle
  )
  ds = ds.prefetch(buffer_size=buffer_size)
  return ds
  
def union_labels(labels):
  return reduce(np.union1d, labels).tolist()

class Dataset:
  def __init__(self, df, labels):
    # Sanity checks!
    if 'path' not in df.columns:
      raise ValueError('Incorrect dataframe format!')
    if not all([l in df.columns for l in labels]):
      raise ValueError('Mismatched labels in dataframe!')
    self.df = df
    self.labels = list(labels)
    # Filter out all unnecessary columns
    self.df = self.df[['path'] + self.labels]
  
  def get_dataset(self, image_shape=(224,224), seed=1337, batch_size=64, buffer_size=32, shuffle=True):
    return create_dataset(self.df, 'path', self.labels, image_shape, seed, batch_size, buffer_size, shuffle)
  
  def get_num_images(self):
    return self.df['path'].count()
  
  def expand_labels(self, labels):
    new_labels = np.setdiff1d(labels, self.labels).tolist()
    expanded_df = self.df.copy()
    expanded_df[new_labels] = 0
    expanded_labels = union_labels([self.labels, new_labels])
    return Dataset(expanded_df, expanded_labels)
  
  @staticmethod
  def merge(dss):
    if not isinstance(dss, (tuple, list, np.ndarray)) or len(dss) <= 1:
      raise ValueError('More than one dataset must be provided!')
    for i, ds in enumerate(dss):
      df = ds.df.copy()
      if i == 0:
        merge_df = df
        merge_labels = np.array(ds.labels)
      else:
        merge_df = pd.concat(
          (merge_df, df),
          ignore_index = True,
          axis = 0
        )
        merge_labels = union_labels([merge_labels, ds.labels])
    merge_df = merge_df[['path'] + merge_labels]
    merge_df[merge_labels] = merge_df[merge_labels].fillna(0).astype(int)
    return Dataset(merge_df, merge_labels)
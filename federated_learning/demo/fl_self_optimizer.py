import os

# 是否开启 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import functools
import attr
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# load dataset
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(True)


# Define preprocessing functions.
def preprocess_fn(dataset, batch_size):
  def batch_format_fn(element):
    return (tf.expand_dims(element['pixels'], -1), element['label'])
  
  return dataset.batch(batch_size).map(batch_format_fn)


# Preprocess and sample clients for prototyping.
train_client_ids = sorted(emnist_train.client_ids)
print(train_client_ids)
train_data = emnist_train.preprocess(preprocess_fn)
central_test_data = preprocess_fn(emnist_train.create_tf_dataset_for_client(train_client_ids[0]))

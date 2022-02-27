"""
联邦平均算法的实现。这旨在成为联合平均的最小独立实现，适用于分支作为算法修改的起点；
根据论文：
Communication-Efficient Learning of Deep Networks from Decentralized Data
H. Brendan McMahan, Eider Moore, Daniel Ramage,Seth Hampson, Blaise Aguera y Arcas.
AISTATS 2017.https://arxiv.org/abs/1602.05629
"""

import tensorflow as tf
import tensorflow_federated as tff
from simple_fedavg_tf import build_server_broadcast_message, client_update, server_update, ServerState


def _initialize_optimizer_vars(model, optimizer):
  """Creates optimizer variables to assign the optimizer's state."""
  model_weights = model.weights
  model_delta = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)
  # Create zero gradients to force an update that doesn't modify.
  # Force eagerly constructing the optimizer variables. Normally Keras lazily
  # creates the variables on first usage of the optimizer. Optimizers such as
  # Adam, Adagrad, or using momentum need to create a new set of variables shape
  # like the model weights.
  grads_and_vars = tf.nest.map_structure(
    lambda x, v: (tf.zeros_like(x), v), tf.nest.flatten(model_delta),
    tf.nest.flatten(model_weights.trainable))
  optimizer.apply_gradients(grads_and_vars)
  assert optimizer.variables()


def build_federated_averaging_process(
    model_fn,
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1)
):
  """Builds the TFF computations for optimization using federated averaging.
  Args:
    model_fn: A no-arg function that returns a
      `simple_fedavg_tf.KerasModelWrapper`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for server update.
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for client update.
  Returns:
    A `tff.templates.IterativeProcess`.
  """
  
  dummy_model = model_fn()
  
  @tff.tf_computation
  def server_init_tf():
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    
    return ServerState(
      model_weights=model.weights,
      optimizer_state=server_optimizer.variables(),
      round_num=0)
  
  server_state_type = server_init_tf.type_signature.result
  
  model_weights_type = server_state_type.model_weights
  
  @tff.tf_computation(server_state_type, model_weights_type.trainable)
  def server_update_fn(server_state, model_delta):
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    
    return server_update(model, server_optimizer, server_state, model_delta)
  
  @tff.tf_computation(server_state_type)
  def server_message_fn(server_state):
    return build_server_broadcast_message(server_state)
  
  server_message_type = server_message_fn.type_signature.result
  tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
  
  @tff.tf_computation(tf_dataset_type, server_message_type)
  def client_update_fn(tf_dataset, server_message):
    model = model_fn()
    client_optimizer = client_optimizer_fn()
    
    return client_update(model, tf_dataset, server_message, client_optimizer)
  
  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_dataset_type = tff.type_at_clients(tf_dataset_type)
  
  @tff.federated_computation(federated_server_state_type, federated_dataset_type)
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.
    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.data.Dataset` with placement
        `tff.CLIENTS`.
    Returns:
      A tuple of updated `ServerState` and `tf.Tensor` of average loss.
    """
    server_message = tff.federated_map(server_message_fn, server_state)
    server_message_at_client = tff.federated_broadcast(server_message)
    
    client_outputs = tff.federated_map(client_update_fn, (federated_dataset, server_message_at_client))
    
    weight_denom = client_outputs.client_weight
    round_model_delta = tff.federated_mean(client_outputs.weights_delta, weight=weight_denom)
    
    server_state = tff.federated_map(server_update_fn,(server_state, round_model_delta))
    round_loss_metric = tff.federated_mean(client_outputs.model_output, weight=weight_denom)
    
    return server_state, round_loss_metric
  
  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    return tff.federated_value(server_init_tf(), tff.SERVER)
  
  return tff.templates.IterativeProcess(initialize_fn=server_init_tff, next_fn=run_one_round)

import torch
import numpy as np

print('-----------initializing a tensor--------------')
# 使用torch.tensor 直接生成 tensor
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# 从 torch.from_numpy 生成一个 tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# from another tensors
# 覆盖原本的模型数据，但是形状，类别不发生变化
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones}")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} ")

# With random or constant values:
# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
print('---------------根据一个shape 可以生成指定 tensor-----------------')
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

print('-----------Attributes of a Tensor----------')
# # Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3, 4)

print(f"Shape of tensor: \n {tensor}")
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

print('--------Operations on Tensors-----------')
# # if tensors are created on the CPU, We need to explicitly move tensors to the GPU using .to method (after checking for GPU availability).
# # We move our tensor to the GPU if available

if torch.cuda.is_available():
  print(f'device: cuda')
  tensor = tensor.to("cuda")
  print(tensor)
else:
  print(f'device: cpu')
  tensor = tensor.to("cpu")
  print(tensor)

# Standard numpy-like indexing and slicing:
print(f"First row:\n {tensor[0]}")
print(f"First column:\n {tensor[:, 0]}")
print(f"Last column: \n {tensor[..., -1]}")
tensor[:, 1] = 0
print(f"tensor[:, 1] = 0:\n {tensor}")

# Joining tensors You can use torch.cat to concatenate a sequence of tensors along a given dimension
t1 = torch.cat([tensor, tensor], dim=1)
print(f"torch.cat([tensor,tensor]):\n {t1}")

# # Arithmetic operation
# # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# y1 = tensor @ tensor.T
# print(f'y1: {y1}')
#
# y2 = tensor.matmul(tensor.T)
# print(f'y2: {y2}')
#
# y3 = torch.rand_like(tensor)
# print(f'y3: {y3}')
#
# y4 = torch.matmul(tensor, tensor.T, out=y3)
# print(f'y4: {y4}')
#
# # This computes the element-wise product. z1, z2, z3 will have the same value
# z1 = tensor * tensor
# print(f'z1: {z1}')
#
# z2 = tensor.mul(tensor)
# print(f'z2: {z2}')
#
# z3 = torch.rand_like(tensor)
# print(f'z3: {z3}')
#
# z4 = torch.mul(tensor, tensor, out=z3)
# print(f'z4 {z4}')

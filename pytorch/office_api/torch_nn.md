# torch.nn api
## Containers


## Convolution Layers
### Conv2d
    参数：
      in_channels (int) – Number of channels in the input image
      out_channels (int) – Number of channels produced by the convolution
      kernel_size (int or tuple) – Size of the convolving kernel
      stride (int or tuple, optional) – Stride of the convolution. Default: 1
      padding (int, tuple or str, optional) – Padding added to all four sides of the input. Default: 0
      padding_mode (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'  
    案例  
      m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
      input = torch.randn(20, 16, 50, 100)
      output = m(input)
      print(f'output shape: {output.shape}')  # torch.Size([20, 33, 28, 100]) 
## Pooling layers

## Padding Layers

## Non-linear Activations (weighted sum, nonlinearity)

## Non-linear Activations (other)

## Normalization Layers

## Recurrent Layers

## Transformer Layers

## Linear Layers

## Dropout Layers
### nn.Dropout(p,inplace)
    作用：按照概率把选择的数值置为：0
    参数  
      p – probability of an element to be zeroed. Default: 0.5
      inplace – If set to True, will do this operation in-place. Default: False

    案例：
      m = nn.Dropout(p=0.2)
      input = torch.randn(4, 4)
      print(f'input content: {input}')
      output = m(input)
      print(f'output after dropout: {output}')

## Sparse Layers

## Distance Functions

## Loss Functions

## Vision Layers

## Shuffle Layers

## DataParallel Layers (multi-GPU, distributed)

## Utilities

## Quantized Functions

## Lazy Modules Initialization
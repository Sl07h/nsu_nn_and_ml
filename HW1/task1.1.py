from math import ceil
from typing import List
import numpy as np
import torch

sample_input = torch.rand(size=[5, 4, 12, 12])  # (N, C, H, W)
sample_weight = torch.rand(size=[16, 4, 3, 3])  # (C_out, C_in, H, W)
# результат должен иметь форму (N, C_out, H, W) = [5, 16, 11, 11]


def conv2d_python(
        image: List[List[List[List[float]]]],
        kernel: List[List[List[List[float]]]],
        stride: int = 1,
        padding_mode: str = 'SAME'):

    n_batches   = len(image)
    C1          = len(image[0])
    height      = len(image[0][0])
    width       = len(image[0][0][0])  # (N, C, H, W)

    n_kernels   = len(kernel)
    C2           = len(kernel[0])
    kernel_size = len(kernel[0][0])

    if C1 != C2:
        RuntimeError('Channel\'s count mismatch')
    C = C1
    
    if padding_mode == 'SAME':  # окна выходят за рамки массива
        padding = int((kernel_size - 1) / 2)
        i_count = ceil(height / stride)
        j_count = ceil(width / stride)
        tensor_image = [[[[0 for j in range(width + kernel_size - 1)]
                          for i in range(height + kernel_size - 1)]
                         for c in range(C)]
                        for n in range(n_batches)]
        # почему-то не работает для list
        # tensor_image[:][:][padding:int(padding+height)][padding:int(padding+width)] = image
        for n in range(n_batches):
            for c in range(C):
                for i1, i2 in enumerate(range(padding, int(padding+height))):
                    for j1, j2 in enumerate(range(padding, int(padding+width))):
                        tensor_image[n][c][i2][j2] = image[n][c][i1][j1]

    elif padding_mode == 'VALID':  # размер массива тот же самый
        i_count = ceil((height - kernel_size + 1) / stride)
        j_count = ceil((width - kernel_size + 1) / stride)
        tensor_image = image

    else:
        print('Wrong padding mode')
        return -1

    res = [[[[0 for j in range(j_count)]
             for i in range(i_count)]
            for n_c in range(n_kernels)]
           for n_b in range(n_batches)]
    for image_index in range(n_batches):
        for kernel_index in range(n_kernels):
            for k_i in range(i_count):
                for k_j in range(j_count):
                    i0 = k_i * stride
                    i1 = i0 + kernel_size
                    j0 = k_j * stride
                    j1 = j0 + kernel_size
                    s = 0.0
                    for channel_index in range(C):
                        for i_kernel, i_image in enumerate(range(i0, i1)):
                            for j_kernel, j_image in enumerate(range(j0, j1)):
                                s += tensor_image[image_index][channel_index][i_image][j_image] * \
                                    kernel[kernel_index][channel_index][i_kernel][j_kernel]
                    res[image_index][kernel_index][k_i][k_j] = s
    print(type(res))
    return res


for padding_torch, padding_my in zip(
    [0, 1],
    ['VALID', 'SAME'],
):
    if not torch.allclose(
        torch.tensor(conv2d_python(sample_input, sample_weight, stride=2, padding_mode=padding_my)),
        torch.nn.functional.conv2d(sample_input, sample_weight, stride=2, padding=padding_torch),
    ):
        raise RuntimeError('You are wrong!')
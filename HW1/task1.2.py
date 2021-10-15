from math import ceil
import numpy as np
import torch

sample_input = torch.rand(size=[5, 8, 25, 25])  # (N, C, H, W)
sample_weight = torch.rand(size=[16, 8, 3, 3])  # (C_out, C_in, H, W)
# результат должен иметь форму (N, C_out, H, W) = [5, 16, 11, 11]


def conv2d_numpy(
        image: np.ndarray,
        kernel: np.ndarray,
        stride: int = 1,
        padding_mode: str = 'SAME'):

    n_batches, C1, height, width = image.shape  # (N, C, H, W)
    n_kernels, C2, kernel_size, _ = kernel.shape
    if C1 != C2:
        RuntimeError('Channel\'s count mismatch')
    C = C1

    if padding_mode == 'SAME':  # окна выходят за рамки массива
        padding = int((kernel_size - 1) / 2)
        i_count = ceil(height / stride)
        j_count = ceil(width / stride)
        tensor_image = np.zeros((n_batches, C, height + kernel_size - 1, width + kernel_size - 1))
        tensor_image[:, :, padding:int(padding+height), padding:int(padding+width)] = image

    elif padding_mode == 'VALID':  # размер массива тот же самый
        i_count = ceil((height - kernel_size + 1) / stride)
        j_count = ceil((width - kernel_size + 1) / stride)
        tensor_image = image

    else:
        print('Wrong padding mode')
        return -1

    res = np.ndarray((n_batches, n_kernels, i_count, j_count), np.float32)
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
                        s += np.sum(tensor_image[image_index, channel_index, i0:i1, j0:j1] * kernel[kernel_index, channel_index])
                    res[image_index][kernel_index][k_i][k_j] = s

    return res


for padding_torch, padding_my in zip(
    [0, 1],
    ['VALID', 'SAME'],
):
    if not torch.allclose(
        torch.tensor(conv2d_numpy(sample_input.numpy(), sample_weight.numpy(), stride=2, padding_mode=padding_my)),
        torch.nn.functional.conv2d(sample_input, sample_weight, stride=2, padding=padding_torch),
    ):
        raise RuntimeError('You are wrong!')
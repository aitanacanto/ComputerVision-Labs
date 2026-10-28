import torch

import numpy as np

def conv1d_one_in_one_out(input_map:np.ndarray, kernel:np.ndarray) -> np.ndarray:
    # Task 1
    # Return a 1D output feature map computed from a 1D input feature map and a 1D single-channel kernel.
    # Formula: y[i] = sum(x[i:i+len(k)] * k)
    
    length = input_map.shape[0] - kernel.shape[0] + 1 # len(input_map) - len(kernel) + 1
    output = np.zeros(length)
    
    for i in range(length):
        output[i] = np.sum(input_map[i:i + kernel.shape[0]] * kernel)
    return output

def conv2d_one_in_one_out(input_map:np.ndarray, kernel:np.ndarray) -> np.ndarray:
    # Task 2
    # Return a 2D output feature map computed from a 2D input feature map and a 2D single-channel kernel.
    # Formula: y[i, j] = np.sum(input_map[i:i+h, j:j+w] * kernel)

    input_h, input_w = input_map.shape  # Imput heigth & weigth 
    kernel_h, kernel_w = kernel.shape   # Kernel heigth & weigth 
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1

    output = np.zeros((output_h, output_w)) # initialazing vector

    for i in range(output_h):
        for j in range(output_w):
            part = input_map[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(part * kernel)
    return output

def conv2d_multiple_in_one_out(input_map:np.ndarray, kernel:np.ndarray) -> np.ndarray:
    # Task 3
    # Return a 2D output feature map computed from multiple 2D input feature maps and a 2D multi-channel kernel.
    
    num_channels, input_h, input_w = input_map.shape # Imput number channels, heigth & weigth  
    _, kernel_h, kernel_w = kernel.shape  # Kernel heigth & weigth
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1

    output = np.zeros((output_h, output_w)) # initialazing vector

    for c in range(num_channels):
        for i in range(output_h):
            for j in range(output_w):
                part = input_map[c, i:i+kernel_h, j:j+kernel_w]
                output[i, j] += np.sum(part * kernel[c])
    return output


def conv2d_multiple_in_multiple_out(input_map:np.ndarray, kernel:np.ndarray) -> np.ndarray:
    # Task 4
    # Return multiple 2D output feature maps computed from multiple 2D input feature maps and multiple 2D multi-channel kernels.
    
    out_channels, in_channels, kernel_h, kernel_w = kernel.shape # Imput channels, output channels, heigth & weigth  
    _, input_h, input_w = input_map.shape  # Kernel heigth & weigth 
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1

    output = np.zeros((out_channels, output_h, output_w)) # initialazing vector

    for oc in range(out_channels):
        for i in range(output_h):
            for j in range(output_w):
                sum_value = 0
                for ic in range(in_channels):
                    part = input_map[ic, i:i+kernel_h, j:j+kernel_w]
                    sum_value += np.sum(part * kernel[oc, ic])
                output[oc, i, j] = sum_value
    return output


def compute_gradient(input_map:np.ndarray, kernel:np.ndarray, target_map:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.tensor(input_map, dtype=torch.float)
    y = torch.tensor(target_map, dtype=torch.float)
    conv2d = torch.nn.Conv2d(kernel.shape[1], kernel.shape[0], (kernel.shape[2], kernel.shape[3]), bias=False)
    conv2d.weight.data = torch.tensor(kernel, dtype=torch.float)
    optimiser = torch.optim.SGD(conv2d.parameters())

    # Task 5
    # Perform forward and  backward propagation with a PyTorch Conv2d module.

    y_pred = conv2d(x)  # Forward, x = Query, Key & Value
    
    loss = torch.sum((y_pred - y) ** 2)  # Loss = (y_pred - y)^2

    optimiser.zero_grad()  # Clean gradients

    loss.backward()  # Backward

    optimiser.step()  # Backward weigth actualization


    gradient = conv2d.weight.grad.detach().numpy()
    new_kernel = conv2d.weight.data.detach().numpy()
    return y_pred.detach().numpy(), gradient, new_kernel


if __name__ == "__main__":

    # Task 1 test code
    print('Task 1:', conv1d_one_in_one_out(np.asarray([0.2, 0.3, 0.4]), np.asarray([1, 2])), sep='\n\n', end='\n\n')


    # Task 2 test code
    print('Task 2:', conv2d_one_in_one_out(np.asarray([[0.2, 0.3, 0.4], [2, 3, 4], [20, 30, 40]]), np.asarray([[1, 2], [2, 1]])), sep='\n\n', end='\n\n')


    # Task 3 test code
    print('Task 3:', conv2d_multiple_in_one_out(np.asarray([[[0.2, 0.3, 0.4], [2, 3, 4], [20, 30, 40]], [[0.4, 0.3, 0.2], [4, 3, 2], [40, 30, 20]]]), np.asarray([[[1, 2], [2, 1]], [[0, -1], [-1, 0]]])), sep='\n\n', end='\n\n')


    # Task 4 test code
    print('Task 4:', conv2d_multiple_in_multiple_out(np.asarray([[[0.2, 0.3, 0.4], [2, 3, 4], [20, 30, 40]], [[0.4, 0.3, 0.2], [4, 3, 2], [40, 30, 20]]]), np.asarray([[[[1, 2], [2, 1]], [[0, -1], [-1, 0]]], [[[-1, -2], [-2, -1]], [[0, 1], [1, 0]]]])), sep='\n\n', end='\n\n')


    # Task 5 test code
    y_pred, gradient, new_kernel = compute_gradient(np.asarray([[[0.2, 0.3, 0.4], [2, 3, 4], [20, 30, 40]], [[0.4, 0.3, 0.2], [4, 3, 2], [40, 30, 20]]]), np.asarray([[[[1, 2], [2, 1]], [[0, -1], [-1, 0]]], [[[-1, -2], [-2, -1]], [[0, 1], [1, 0]]]]), np.zeros((2, 2, 2)))
    print('Task 5:', 'output:', y_pred, 'gradient:', gradient, 'new kernel:', new_kernel, sep='\n\n', end='\n\n')
    
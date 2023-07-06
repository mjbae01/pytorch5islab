from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

from torch4is.utils import time_log
from torch4is.my_nn.linear import MyLinear

def generate_data() -> Tuple[torch.Tensor, torch.Tensor]:
    weight = np.array([[0.5], [-1.5]])
    bias = np.array([-0.05])

    x = np.random.uniform(-1, 1, (10000,2)).astype(np.float32)
    y_answer = np.dot(x, weight)+bias
    y_noisy_answer = y_answer + 0.01 * np.random.randn(*y_answer.shape)
    #y_answer: 가변인자, 몇개가 들어올 지 모름, y_answer 크기만큼 random하게 생성

    x_t = torch.from_numpy(x) #tensor로 변환, 메모리를 공유
    y_t = torch.from_numpy(y_noisy_answer)
    return x_t, y_t

def create_network() -> nn.Module:
    net = MyLinear(2,1) #input dim:2 , output dim:1 선형변환 수행

    print(time_log())
    for param_name, param in net.named_parameters():
        print(f"Initial parameter ({param_name}): {param} (shape: {param.shape}, num_elements: {param.numel()})")
    return net

def fit(net: nn.Module,
        data_x : torch.Tensor,
        data_y : torch.Tensor,
        max_iters: int = 100,
        w_lr: float = 0.01,
        b_lr: float = 0.0001) -> None:
    print(time_log())
    for num_iter in range(max_iters):
        net.zero_grad(set_to_none=True)
        # set_to_none=True : 메모리 사용량이 작아져 퍼포먼스가 소폭 향상됨
        pred_y = net(data_x)

        loss = torch.sum(torch.square(pred_y - data_y))
        loss.backward()

        #
        with torch.no_grad():
            for param in net.parameters():
                if param.grad is None:
                    continue
                if param.ndim == 1: #bias
                    param -= b_lr * param.grad
                else: #weight
                    param -= w_lr * param.requires_grad

        print(f"... iter {num_iter} / {max_iters}, loss: {loss.item()}")

    print(time_log())
    for param_name, param in net.named_parameters():
        print(f"Final parameter ({param_name}): {param}")

def run():
    data_x, data_y = generate_data()
    net = create_network()

    fit (net, data_x, data_y, max_iters = 100, w_lr=1e-4, b_lr=1e-6)

if __name__ == '__main__':
    run()



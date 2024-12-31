import torch

@torch.no_grad()
def acc_top1(y:torch.Tensor, y_hat:torch.Tensor):
    pre = torch.argmax(y_hat, dim=1)
    pre_result = y == pre
    acc = torch.sum(pre_result) / len(y)
    return acc.item()

@torch.no_grad()
def acc_topk(y:torch.Tensor, y_hat:torch.Tensor, k=3):
    y = y.cpu()
    y_hat = y_hat.cpu()
    topk = torch.topk(y_hat, k, dim=1)[1]
    y = y.reshape(-1, 1).repeat(1, k)
    pre_result = y == topk
    pre_result = pre_result.any(dim=1)
    
    acc = torch.sum(pre_result) / len(y)
    return acc.item()

if __name__ == '__main__':
    pass

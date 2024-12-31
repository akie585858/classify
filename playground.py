import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = torch.load('result/dense_bc190_cifar100_argument/train_result.pt')
    test_data = torch.load('result/dense_bc190_cifar100_argument/test_result.pt')

    print(len(data))
    # exit()
    loss = [d['acc_1'] for d in data]
    test_loss = [d['acc_1'] for d in test_data]
    test_x = torch.arange(1, len(loss)+1, 3)
    
    plt.title('loss curve')
    plt.plot(loss, 'o-', linewidth=2, label='train_loss',)
    plt.plot(test_x.tolist(), test_loss, 'o-', linewidth=2, label='test_loss',)
    plt.legend()
    plt.grid()
    plt.show()

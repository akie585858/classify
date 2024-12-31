from mylib.utils.recorder import Counter, Saver
from mylib.utils.trianner import Trainner
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch
import os
import json
import copy
from torchvision import transforms

from model.DenseNet import dense_bc_img32_cls100
from dataset.cifar100 import CIFAR100
from tagetFn.accuracy import acc_topk


def process_setting(json_file):
    with open(json_file, 'r') as f:
        json_result = json.load(f)

    train_set = json_result['train']

    # 参数配置
    save_root = train_set['save_root']
    eval_attr = train_set['eval_attr']
    pretrain_file = train_set['pretrain_file']
    batch_size = train_set['batch_size']
    real_batch = train_set['real_batch']

    scheduler = train_set['scheduler']
    if scheduler['fit'] is None:
        fit_lr_scheduler = None
        fit_init_lr = None
    else:
        fit = scheduler['fit']
        fit_init_lr = fit['lr']
        fit_lr_scheduler = {'epochs_num':fit['epochs_num'], 'stones':fit['stones'], 'gamma':fit['gamma']}

    if scheduler['warm'] is None:
        warm_lr_scheduler = None
    else:
        warm = scheduler['warm']
        warm_lr_scheduler = {'epochs_num':warm['epochs_num'], 'stones':warm['stones'], 'gamma':warm['gamma']}

    train = scheduler['train']
    train_init_lr = train['lr']
    train_lr_scheduler = {'epochs_num':train['epochs_num'], 'stones':train['stones'], 'gamma':train['gamma']}

    # 根据记录文件调整lr及策略
    if os.path.exists(os.path.join(save_root, 'last.tmp')):
        with open(os.path.join(save_root, 'last.tmp'), 'r') as f:
            complete_nums = int(f.readline())
            best_val = float(f.readline())
        if not fit_lr_scheduler is None:
            fit_epochs_num = fit_lr_scheduler['epochs_num']

            if complete_nums - fit_epochs_num >= 0:
                complete_nums -= fit_epochs_num
                fit_lr_scheduler = None
            else:
                fit_lr_scheduler['epochs_num'] -= complete_nums
                lr = fit_init_lr
                stones = copy.deepcopy(fit_lr_scheduler['stones'])
                for stone in fit_lr_scheduler['stones']:
                    if complete_nums - stone >= 0:
                        lr *= fit_lr_scheduler['gamma']
                        stones.pop(0)
                    else:
                        stones = [i - complete_nums for i in stones]
                        break
                if stones == []:
                    stones = None
                fit_lr_scheduler['stones'] = stones
                fit_init_lr = lr

        if not warm_lr_scheduler is None:
            warm_epochs_num = warm_lr_scheduler['epochs_num']

            if complete_nums - warm_epochs_num >= 0:
                warm_lr_scheduler = None
                complete_nums -= warm_epochs_num
            else:
                warm_lr_scheduler['epochs_num'] -= complete_nums
                stones = copy.deepcopy(warm_lr_scheduler['stones'])
                for stone in warm_lr_scheduler['stones']:
                    if complete_nums - stone >= 0:
                        stones.pop[0]
                    else:
                        stones = [i - complete_nums for i in stones]
                        break

        if not train_lr_scheduler is None:
            train_epochs_num = train_lr_scheduler['epochs_num']

            if complete_nums - train_epochs_num >= 0:
                # 该任务已完成
                exit()
            else:
                train_lr_scheduler['epochs_num'] -= complete_nums
                lr = train_init_lr
                stones = copy.deepcopy(train_lr_scheduler['stones'])
                for stone in train_lr_scheduler['stones']:
                    if complete_nums - stone >= 0:
                        lr *= train_lr_scheduler['gamma']
                        stones.pop(0)
                    else:
                        stones = [i - complete_nums for i in stones]
                        break
                if stones == []:
                    stones = None
                train_lr_scheduler['stones'] = stones
                train_init_lr = lr

    else:
        complete_nums=0
        best_val = None

    return complete_nums, best_val, pretrain_file, save_root, eval_attr, batch_size, real_batch, train_init_lr, train_lr_scheduler, fit_init_lr, fit_lr_scheduler, warm_lr_scheduler

if __name__ == '__main__':
    json_file = 'result/dense_bc190_cifar100_argument/setting.json'
    complete_nums, best_val, pretrain_file, save_root, eval_attr, batch_size, real_batch, train_init_lr, train_lr_scheduler, fit_init_lr, fit_lr_scheduler, warm_lr_scheduler = process_setting(json_file)

    # 手动模型配置-------------------------------------------------------------------------
    # 配置模型
    net = dense_bc_img32_cls100(190, 40)
    if not pretrain_file is None:
        net.load_state_dict(torch.load(pretrain_file))
    elif os.path.exists(os.path.join(save_root, 'last.pt')):
        net.load_state_dict(torch.load(os.path.join(save_root, 'last.pt')))
    loss_fn = nn.CrossEntropyLoss()

    # 配置数据集
    data_argument = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4)
    ])

    train_dataset = CIFAR100(data_argument=data_argument)
    test_dataset = CIFAR100(False)

    # 配置计数器, 保存器
    train_counter = Counter(
        loss=lambda loss:loss.item(),
        acc_1=lambda y, y_hat:acc_topk(y, y_hat, 1),
        acc_3=lambda y, y_hat:acc_topk(y, y_hat, 3)
    )

    test_counter = Counter(
        loss=lambda loss:loss.item(),
        acc_1=lambda y, y_hat:acc_topk(y, y_hat, 1),
        acc_3=lambda y, y_hat:acc_topk(y, y_hat, 3)
    )   

    saver = Saver(
        save_root=save_root,
        eval_attr=eval_attr,
        test_rate=3,
        epochs_num=complete_nums,
        best_val=best_val
    )

    # 配置训练器
    trianner = Trainner(
        net=net,
        loss_fn=loss_fn,
        dataset=train_dataset,
        test_dataset=test_dataset,
        counter = train_counter,
        test_counter=test_counter,
        batch_size=batch_size,
        r_batch_size=real_batch,
        saver=saver
    )

    # 完成配置----------------------------------------------------------------------------
    # 生成schedul_list
    schedul_list = []

    train_optim = SGD(net.parameters(), lr=train_init_lr, weight_decay=1e-4, momentum=0.9, nesterov=True, dampening=False)
    train_scheduler = MultiStepLR(train_optim, train_lr_scheduler['stones'], train_lr_scheduler['gamma'])

    if not fit_lr_scheduler is None:
        fit_optim = SGD(net.parameters(), lr=fit_init_lr)
        fit_scheduler = MultiStepLR(fit_optim, fit_lr_scheduler['stones'], fit_lr_scheduler['gamma'])
        schedul_list.append({
            'optim':fit_optim,
            'lr_scheduler':fit_scheduler,
            'epochs_num':fit_lr_scheduler['epochs_num'],
            'describe':'fit'
        })
    if not warm_lr_scheduler is None:
        warm_scheduler = MultiStepLR(train_optim, warm_lr_scheduler['stones'], warm_lr_scheduler['gamma'])
        schedul_list.append({
            'optim':train_optim,
            'lr_scheduler':train_scheduler,
            'epochs_num':warm_lr_scheduler['epochs_num'],
            'describe':'warm'
        })

    schedul_list.append({
            'optim':train_optim,
            'lr_scheduler':train_scheduler,
            'epochs_num':train_lr_scheduler['epochs_num'],
            'describe':'train'
        })

    trianner.train(
        test_rate=3,
        schedul_list=schedul_list
    )


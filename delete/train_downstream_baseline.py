'''
Author: sy
Date: 2021-04-06 09:18:56
LastEditors: your name
LastEditTime: 2021-11-01 14:48:46
Description: 使用训练好的s3net的backbone来训练属性识别任务
'''

#coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.optim import lr_scheduler
import datetime
from tensorboardX import SummaryWriter as sw

from utils import cal_attr_acc, lambda_warm_up, lambda_warm_up_2
from configs import cfg, show_cfg
from datasets import make_dataloader
from models import get_down_model, PreActResNet18

def train_downstream_method(gpu_id, cfg_pth):
    parser = argparse.ArgumentParser(description="S3NET  Downstream Training")
    parser.add_argument("--config_file", default=cfg_pth, help="path to config file", type=str)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    writer = sw(cfg.SW_LOG_DIR)
    max_acc = 0
    max_epoch = 0

    #======================= load data and model =======================
    trainloader, valloader, testloader = make_dataloader(cfg, group=[12, 13, 6, 9])
    print_inter = int(len(trainloader) // cfg.PRINT_INTER)
    if print_inter == 0:
        print_inter = 1

    model = PreActResNet18(40) # get_down_model(num_classes_cls=[12, 13, 6, 9])
    
    if cfg.TRAIN.IS_CONTINUE:
        model_path = cfg.TRAIN.LOAD_PTH + 'epoch{}.pth'.format(cfg.TRAIN.LOAD_EPOCH)
        model.load_state_dict(torch.load(model_path))
    elif cfg.TRAIN.IS_PRETRAINED:
        model_path = cfg.TRAIN.PRETRAIN_PTH + 'epoch{}.pth'.format(cfg.TRAIN.PRETRAIN_EPOCH)
        model.load_param(model_path)
    else:
        model_path = 'baseline'
    print('Model loaded from {}......'.format(model_path))

    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()

    #======================= set loss and optimizer ====================
    criterion = nn.BCEWithLogitsLoss() # 还没测试
    # criterion = BalancedLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=cfg.TRAIN.LR, betas=(0.9, 0.999),weight_decay=5e-5)
    scheduler=lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_warm_up_2)
    

    #====================== show info =================================
    show_cfg(cfg)
    print('\nusing dataset [{}], select rate: {}'.format(cfg.DATA.DATASET_NAME,cfg.DATA.SELECT_RATE))  
    print('trainloader: ',len(trainloader))  
    print('testloader: ',len(testloader))
    print('print every {} batch'.format(print_inter))

    #====================== train start ===============================
    iter_counter=0
    
    for epoch in range(0, cfg.TRAIN.N_EPOCH):
        #-------------------------------- train -------------------------
        #----------------------------------------------------------------
        model.train()
        start_time=datetime.datetime.now()
        print('\nTrain epoch %d' % epoch,' at',start_time )
        current_lr=optimizer.state_dict()['param_groups'][0]['lr']
        print('current lr: %f' % current_lr)

        for batch_idx, (imgs, attrs) in enumerate(trainloader):
            iter_counter += 1
            optimizer.zero_grad()

            images = imgs.cuda()
           
            attrs = attrs.cuda().type(torch.cuda.FloatTensor)
            output = model(images)
            # print(output)
            # print(attrs)

            
            
            loss = criterion(output, attrs)
            
                
            loss.backward()
            optimizer.step()

            # print_inter = 1
            if batch_idx % int(print_inter) == 0:
                correct = torch.FloatTensor(40).fill_(0)
                total = 0
                with torch.no_grad():
                   

                    c, t = cal_attr_acc(output.cpu().data, attrs.cpu().data, 40)
                    correct.add_(c)
                    total += t
                train_acc = torch.mean(correct / total)
                
                print('[%d/%d][%d/%d]: train_loss: %.4f, train_acc: %.4f'
                      % (epoch, cfg.TRAIN.N_EPOCH, batch_idx, len(trainloader), loss.item(), train_acc))
                
                # iter_counter = len(trainloader) + batch_idx
                writer.add_scalar('train_loss', loss.item(), iter_counter)
                writer.add_scalar('train_acc', train_acc, iter_counter)

            # if batch_idx == 0:
            #     break

        end_time1=datetime.datetime.now()
        print('train run time %.2f min' % ((end_time1 - start_time).total_seconds() / 60 + 1))
        
        #---------------------------- eval ----------------------------------
        #--------------------------------------------------------------------
        print('| Eval mode......')
        model.eval()
        print('| current lr: %f' % current_lr)

        with torch.no_grad():
            correct = torch.FloatTensor(40).fill_(0)
            total = 0
            for batch_idx, (images, attrs) in enumerate(testloader):
                
                images = images.cuda()
                
                attrs = attrs.cuda().type(torch.cuda.FloatTensor)
                output = model(images)
                # print(output)
                # print(attrs)
               

                c, t = cal_attr_acc(output.cpu().data, attrs.cpu().data, 40)
                correct.add_(c)
                total += t

                # if batch_idx == 0:
                #     break
        scheduler.step()
          
        print('| current accuracy per attr:\n',correct / total)
        mean_acc=torch.mean(correct/total)
        print('| current mean accuracy: ',mean_acc)
        writer.add_scalar('eval_accuracy',mean_acc,epoch)
        writer.add_text('record per attr eval_acc', str(correct / total) + ' current mean accuracy: ' + str(mean_acc), epoch)
        
        if max_acc < torch.mean(correct / total):
            max_acc = torch.mean(correct / total)
            max_epoch = epoch
        print('| max mean accuracy: %.4f at epoch %d' % (max_acc, max_epoch))
        writer.add_text('record max', ('max acc:') + str(max_acc) + ' AT epoch:' + str(max_epoch), epoch)
        

        #=========================== save model ============================
        if epoch == max_epoch:
            save_pth = cfg.TRAIN.SAVE_PTH + 'epoch{}_{}.pth'.format(epoch, mean_acc)
            if not os.path.exists(cfg.TRAIN.SAVE_PTH):
                os.makedirs(cfg.TRAIN.SAVE_PTH)
            torch.save(model.state_dict(), save_pth)  # local
            print('Model saved in {}......'.format(save_pth))

        # if epoch == 0:
        #     break
        
    writer.add_text('cfg',str(cfg))      
    writer.close()

if __name__ == '__main__':
    cfg_pth='/home/sy/s3net/config_files/train_downstream.yml'
    train_downstream_method('0', cfg_pth)

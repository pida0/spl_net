

#coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from torch.optim import lr_scheduler
import datetime
from tensorboardX import SummaryWriter as sw
from models.trans_s3net_5b import get_down_model_5b

from utils import cal_attr_acc, lambda_warm_up_2
from configs import cfg, show_cfg
from datasets import make_dataloader

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

    model = get_down_model_5b()
    
    if cfg.TRAIN.IS_CONTINUE:
        model_path = cfg.TRAIN.LOAD_PTH + 'epoch{}.pth'.format(cfg.TRAIN.LOAD_EPOCH)
        model.load_state_dict(torch.load(model_path))
    elif cfg.TRAIN.IS_PRETRAINED:
        model_path = cfg.TRAIN.PRETRAIN_PTH + 'epoch{}.pth'.format(cfg.TRAIN.PRETRAIN_EPOCH)
        model.load_param(model_path)
    else:
        model_path = 'baseline'
    print('Model loaded from {}......'.format(model_path))

    model.cuda()

    #======================= set loss and optimizer ====================
    criterion = nn.BCEWithLogitsLoss() 
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
            for i in range(4):
                attrs[i] = attrs[i].cuda().type(torch.cuda.FloatTensor)
            output = model(images)


            loss = [0] * 4
            total_loss = 0
            for i in range(4):
                loss[i] = criterion(output[i], attrs[i])
                total_loss += loss[i]
                
            total_loss.backward()
            optimizer.step()

            if batch_idx % int(print_inter) == 0:
                correct = torch.FloatTensor(40).fill_(0)
                total = 0
                with torch.no_grad():
                    pred = torch.cat(output, 1)
                    label = torch.cat(attrs, 1)

                    c, t = cal_attr_acc(pred.cpu().data, label.cpu().data, 40)
                    correct.add_(c)
                    total += t
                train_acc = torch.mean(correct / total)
                
                print('[%d/%d][%d/%d]: train_loss: %.4f, whole_loss: %.4f, upper_loss: %.4f, middle_loss: %.4f, lower_loss: %.4f, train_acc: %.4f'
                      % (epoch, cfg.TRAIN.N_EPOCH, batch_idx, len(trainloader), total_loss.item(), loss[0].item(),
                         loss[1].item(), loss[2].item(), loss[3].item(), train_acc))
                
                writer.add_scalar('train_loss', total_loss.item(), iter_counter)
                writer.add_scalar('train_acc', train_acc, iter_counter)


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
                for i in range(4):
                    attrs[i] = attrs[i].cuda().type(torch.cuda.FloatTensor)
                output = model(images)

                pred = torch.cat(output, 1)
                label = torch.cat(attrs, 1)

                c, t = cal_attr_acc(pred.cpu().data, label.cpu().data, 40)
                correct.add_(c)
                total += t

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
            save_pth = cfg.TRAIN.SAVE_PTH + 'epoch{}.pth'.format(epoch)
            if not os.path.exists(cfg.TRAIN.SAVE_PTH):
                os.makedirs(cfg.TRAIN.SAVE_PTH)
            torch.save(model.state_dict(), save_pth)  # local
            print('Model saved in {}......'.format(save_pth))

        
    writer.add_text('cfg',str(cfg))      
    writer.close()

if __name__ == '__main__':
    cfg_pth='./config_files/train_downstream.yml'
    train_downstream_method('0', cfg_pth)


#coding: utf-8
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import argparse
import datetime

from torch.utils.tensorboard  import SummaryWriter as sw
from torch.optim import lr_scheduler
import numpy as np
import warnings
warnings.filterwarnings("ignore")


from configs import cfg, show_cfg
from models import get_model_5b, Discriminators
from utils import record, show_eval_result, lambda_warm_up, CrossEntropyLabelSmooth
from datasets import make_dataloader


def train_pretext_method(gpu_id, cfg_pth):
    #======================= load configs ===============================
    parser = argparse.ArgumentParser(description="S3NET Training")
    parser.add_argument("--config_file", default=cfg_pth, help="path to config file", type=str)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.freeze()

    
    writer = sw(cfg.SW_LOG_DIR)
    max_acc = [0, 0, 0, 0, 0] # |seg_acc-|-seg_mIoU-|-self_cls_acc-|-rot_acc-|-rot_angle_acc-|
    max_epoch = [0, 0, 0, 0, 0]
    

    #======================= load data and model =======================
    trainloader, valloader, testloader = make_dataloader(cfg, group=[2, 7, 4, 6])
    print_inter = int(len(trainloader) // cfg.PRINT_INTER)
    if print_inter == 0:
        print_inter = 1

    model = get_model_5b(num_classes_rot=cfg.DATA.PATCH_NUM,
                      num_classes_seg=19)
    net_D = Discriminators()

    if cfg.TRAIN.IS_CONTINUE:
        model_path = cfg.TRAIN.LOAD_PTH + 'epoch{}.pth'.format(cfg.TRAIN.LOAD_EPOCH)
        model.load_state_dict(torch.load(model_path))
        print('Model loaded from {}......'.format(model_path))

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    model.cuda()
    net_D.cuda()

    #======================= set loss and optimizer ====================
    pst_criterion = CrossEntropyLabelSmooth(19) # pst uses label smooth
    pct_criterion = nn.BCEWithLogitsLoss()
    prt_cls_criterion = nn.CrossEntropyLoss()
    prt_adv_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=cfg.TRAIN.LR, betas=(0.9, 0.999), weight_decay=5e-4)
    d_optimizer = optim.Adam(net_D.parameters(), lr=cfg.TRAIN.LR, betas=(0.9, 0.99), weight_decay=5e-4)
    
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_warm_up)

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
        net_D.train()
        start_time=datetime.datetime.now()
        print('\nTrain epoch %d' % epoch,' at',start_time )
        current_lr=optimizer.state_dict()['param_groups'][0]['lr']
        print('current lr: %f' % current_lr)


        for batch_idx, (imgs, labels) in enumerate(trainloader):
            iter_counter += 1
            
            for i in range(4):
                if i==0 or i==1:
                    imgs[i] = imgs[i].cuda()
                if i == 1:
                    for j in range(4):
                        labels[i][j] = labels[i][j].cuda()
                else:
                    labels[i] = labels[i].cuda()

            prt_out, prt_angle_out, pst_out, pct_out, x_prtg_fake, x_prtg_real = model(imgs[0], imgs[1])
            x_prtg_real_logit = x_prtg_real.detach()

            pct_loss = 0
            for i in range(4):
                pct_loss += pct_criterion(pct_out[i], labels[1][i])
                
            prt_cls_loss = prt_cls_criterion(prt_out, labels[2])
            prt_rot_loss = prt_cls_criterion(prt_angle_out, labels[3])
            

            if batch_idx % cfg.TRAIN.D_REPEAT == 0:
                # Compute loss with predict feat
                out_d_fake = net_D(x_prtg_fake)
                d_pred_loss = prt_adv_criterion(out_d_fake, torch.zeros(out_d_fake.size()).cuda())

                # Compute loss with groundtruth feat
                out_d_real = net_D(x_prtg_real_logit)
                d_gt_loss = prt_adv_criterion(out_d_real, torch.ones(out_d_real.size()).cuda())
                
                # Backward and optimize
                d_loss = (d_pred_loss + d_gt_loss) * 0.5
                d_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                d_optimizer.step()    
            out_d_fake = net_D(x_prtg_fake)
            prt_adv_loss = prt_adv_criterion(out_d_fake, torch.ones(out_d_fake.size()).cuda())

            pst_out = F.interpolate(input=pst_out, size=(224, 224), mode='bilinear', align_corners=True)
            pst_loss = pst_criterion(pst_out, labels[0])

            loss = pst_loss + pct_loss + prt_cls_loss + prt_adv_loss + prt_rot_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % (print_inter) == 0:
                print('[%d/%d][%d/%d]' % (epoch, cfg.TRAIN.N_EPOCH, batch_idx, len(trainloader)))
                print('[Loss]: pst_loss: %.4f, pct_loss: %.4f, prt_cls_loss: %.4f, prt_adv_loss: %.4f, prt_rot_loss: %.4f, total_loss: %.4f'
                      % (pst_loss, pct_loss, prt_cls_loss, prt_adv_loss, prt_rot_loss, loss))
                
                writer.add_scalar('train_loss', loss.item(), iter_counter)
                with torch.no_grad():
                    record(prt_out, prt_angle_out, pst_out, pct_out, labels, writer, iter_counter, 'train')
            
        end_time1=datetime.datetime.now()
        print('train run time %.2f min' % ((end_time1 - start_time).total_seconds() / 60 + 1))
        
        #---------------------------- eval ----------------------------------
        #--------------------------------------------------------------------
        print('| Eval mode......')
        model.eval()
        net_D.eval()
        acc = np.array([0, 0, 0, 0, 0],dtype='float64')
        cnt = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(testloader):
                cnt += 1
                for i in range(4):
                    if i==0 or i==1:
                        imgs[i] = imgs[i].cuda()
                    if i == 1:
                        for j in range(4):
                            labels[i][j] = labels[i][j].cuda()
                    else:
                        labels[i] = labels[i].cuda()
                    
                
                prt_out, prt_angle_out, pst_out, pct_out, _, _ = model(imgs[0], imgs[1])
                pst_out = F.interpolate(input=pst_out, size=(224, 224), mode='bilinear', align_corners=True)

                cur_acc = record(prt_out, prt_angle_out, pst_out, pct_out, labels, writer, iter_counter, 'eval')
                acc += cur_acc

        scheduler.step()

        end_time2 = datetime.datetime.now()
        print('| eval run time %.2f min' % ((end_time2 - end_time1).total_seconds() / 60 + 1))

        mean_acc = acc / cnt
        show_eval_result(mean_acc, max_acc, max_epoch, epoch, writer)

        #=========================== save model ============================
        save_pth = cfg.TRAIN.SAVE_PTH + 'epoch{}.pth'.format(epoch)
        if not os.path.exists(cfg.TRAIN.SAVE_PTH):
            os.makedirs(cfg.TRAIN.SAVE_PTH)
        torch.save(model.state_dict(), save_pth)  # loca
        print('Model saved in {}......'.format(save_pth))

        
    writer.add_text('record self max',('max acc:')+str(max_acc)+' AT epoch:'+str(max_epoch))
    writer.close()


if __name__ == '__main__':
    cfg_pth='./config_files/test.yml'
    train_pretext_method('2', cfg_pth)

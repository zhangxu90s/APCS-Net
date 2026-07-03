import torch
import torch.nn as nn
import math
from utils import gm, rand_bbox, bce_loss, pseudo_gtmask, balanced_mask_loss_ce, ngwp_focal,  ClassAwareTripletLoss, BCEWithLogitsLossWithWeights, BCEWithLogitsLossWithIgnoreIndex, attention_cam
from wss import *
import torch.nn.functional as F
import torchvision
import numpy as np
import argparse
from segm.optim.factory import create_optimizer, create_scheduler
import os

def token_masking_2d(tokens, mask_ratio=0.2):
    """
    tokens: [B, H, W, C]
    mask_ratio: 丢弃 token 比例
    """
    B, H, W, C = tokens.shape

    # 随机生成 mask
    mask = (torch.rand(B, H, W, device=tokens.device) > mask_ratio).float()
    mask = mask.unsqueeze(-1)  # [B,H,W,1]

    return tokens * mask

def stain_shift(img):
    """
    img: [B,3,H,W]
    """

    scale = torch.empty(
        img.size(0),3,1,1,
        device=img.device
    ).uniform_(0.8,1.2)

    bias = torch.empty(
        img.size(0),3,1,1,
        device=img.device
    ).uniform_(-0.1,0.1)

    img = img * scale + bias

    return img.clamp(0,1)
    
def ngwp_focal(outputs, beta=1, lam=1e-2):
    """
    outputs: 输入张量，通常是卷积层的输出特征图，维度为 [bs, c, h, w]
    beta:    用于平衡 NGWP 项和 Focal 项的权重系数
    lam:     平滑项，防止 log 函数输入为 0
    """
    bs, c, h, w = outputs.size()

    masks = F.softmax(outputs, dim=1)
    
    masks_ = masks.view(bs, c, -1)

    logits = outputs.view(bs, c, -1)
    
    
    y_ngwp = (logits * masks_).sum(-1) / (1.0 + masks_.sum(-1))

    y_focal = torch.pow(1 - masks_.mean(-1), 3) * torch.log(lam + masks_.mean(-1))
    
    
    y = y_ngwp + beta * y_focal 

    return y





def train(path_work, model, dataloader_train, device, hp, valid_fn=None, dataloader_valid=None, test_num_pos=0, args=None):
    
    r = hp['r']
    lr = hp['lr']
    wd = hp['wd']
    num_epoch = hp['epoch']
    start_epoch = hp['start_epoch']
    best_result = 0
    print('Learning Rate: ', lr)
    loss_fn = nn.BCELoss()
    criterion = BCEWithLogitsLossWithIgnoreIndex(reduction='none')
    dataset_size = len(dataloader_train.dataset)

    if hp['optimizer'] == 'side':
        params1 = list(map(id, model.decoder1.parameters()))
        params2 = list(map(id, model.decoder2.parameters()))
        params3 = list(map(id, model.decoder3.parameters()))
        # params4 = list(map(id, model.linear_fuse.parameters()))
        base_params = filter(lambda p: id(p) not in params1 + params2 + params3, model.parameters())
        params = [{'params': base_params},
                  {'params': model.decoder1.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': model.decoder2.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': model.decoder3.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                #   {'params': model.linear_fuse.parameters(), 'lr': lr / 100, 'weight_decay': wd}
                  ]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    
    elif hp['optimizer'] == 'sgd':
        # optimizer
        optimizer_kwargs = dict(
            opt='sgd',
            lr=lr,
            weight_decay=0.0,
            momentum=0.9,
            clip_grad=None,
            sched='polynomial',
            epochs=num_epoch,
            min_lr=1e-5,
            poly_power=0.9,
            poly_step_size=1,
        )
        print(optimizer_kwargs)
        optimizer_kwargs["iter_max"] = len(dataloader_train) * optimizer_kwargs["epochs"]
        optimizer_kwargs["iter_warmup"] = 0.0
        opt_args = argparse.Namespace()
        opt_vars = vars(opt_args)
        for k, v in optimizer_kwargs.items():
            opt_vars[k] = v
        optimizer = create_optimizer(opt_args, model)
        lr_scheduler = create_scheduler(opt_args, optimizer)
    
    print("{:*^50}".format("training start"))
    for epoch in range(start_epoch, num_epoch):
        epoch_loss = 0
        step = 0
        batch_num = len(dataloader_train)
        num_updates = epoch * len(dataloader_train)

        for index, batch in enumerate(dataloader_train):
            image, label, other = batch
        
            img_show = other["img_show"]
            image = image.to(device)
            label = label.to(device)
            img_show = img_show.to(device)
            bs = image.shape[0]
             
     
            im_aug = stain_shift(image)
                    
            model.train()
      

            if args.model == 'segmenter':

                output = model(image)

                fin_masks = output["fin_patch_masks"]
                y = ngwp_focal(fin_masks)

                loss = loss_fn(torch.sigmoid(y), label)
                
                with torch.no_grad():
                    pseudo = torch.sigmoid(y).detach()

                output_aug = model(im_aug)

                fin_masks_aug = output_aug["fin_patch_masks"]
                y_aug = ngwp_focal(fin_masks_aug)

                loss_cons = F.binary_cross_entropy(
                    torch.sigmoid(y_aug), pseudo
                )

                loss = loss + loss_cons
                
                
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            if args.model == 'segmenter':
                num_updates += 1
                lr_scheduler.step_update(num_updates=num_updates)

            epoch_loss += loss.item()
            step += 1
            
            if index % 100 == 0:
                print("batch %d/%d loss:%0.4f" % (index, batch_num, loss.item()))
        
        epochs = epoch + 1
        average_loss = epoch_loss / math.ceil(dataset_size // dataloader_train.batch_size)
        print("epoch %d loss:%0.4f" % (epochs, average_loss))

        state = {"model_state": model.state_dict(), 
                "epoch": epochs}
        
        if valid_fn is not None:
            model.eval()
            result = valid_fn(model, dataloader_valid, test_num_pos, device, args)
            print('epoch %d loss:%.4f result:%.3f' % (epochs, average_loss, result))
            print(result, best_result)
            if result > best_result:
                best_result = result
                ckpt_name = 'best_model.pth'
                if args.ckpt_name:
                    ckpt_name = f'best_model_{args.ckpt_name}.pth'
                torch.save(state, path_work + ckpt_name)
        
        ckpt_name = 'final_model.pth'
        if args.ckpt_name:
            ckpt_name = f'final_model_{args.ckpt_name}.pth'
        
    torch.save(state, os.path.join(path_work, ckpt_name))
    
    print('best result: %.3f' % best_result)

def binarize(input):
    max = input.max(dim=1, keepdim=True)[0]
    return (input >= max).type_as(input)
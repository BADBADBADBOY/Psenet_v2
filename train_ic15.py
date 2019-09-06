import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.autograd import Variable
from torch.utils import data
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] ="3"
from dataset import IC15Loader
from metrics import runningScore
import models
import torchvision
from util import Logger, AverageMeter
import time
import util

def get_label_num(text,min_area):
    label_num, label = cv2.connectedComponents(text, connectivity=4)
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            label_num-=1
    return label_num,label

def get_cat_tag(outputs,gt_text,gt_kernel):
    outputs = (torch.sign(outputs - 1.0) + 1) / 2
    ind = 'ori'
    cv2.imwrite('text'+str(ind)+'.jpg',outputs[0, 0, :, :].data.cpu().numpy().astype(np.uint8)*255)
    cv2.imwrite('kernel'+str(ind)+'.jpg',outputs[0, 1, :, :].data.cpu().numpy().astype(np.uint8)*255)
    
    text =gt_text #outputs[:, 0, :, :]*gt_text #
    kernel =gt_kernel[0] #outputs[:, 1, :, :] * gt_kernel #

    text = text.data.cpu().numpy().astype(np.uint8)
    kernel = kernel.data.cpu().numpy().astype(np.uint8)
   
    label_num_text, label_text = get_label_num(text, 400)
    label_num_kernel, label_kernel = get_label_num(kernel, 400)
    
    ind = 'select'
    cv2.imwrite('text'+str(ind)+'.jpg',label_text.astype(np.uint8)*255)
    cv2.imwrite('kernel'+str(ind)+'.jpg',label_kernel.astype(np.uint8)*255)

    tag = []
    for i in range(1, int(label_kernel.max()) + 1):
        if ((label_kernel == i).sum() > 0):
            tag.append(i)
    tag_cat = []
    for i in tag:
        tag_cat.append([i, ((label_kernel == i) * label_text).max()])
    return tag_cat,label_kernel,label_text

def get_batch_tag(outputs,batch,gt_texts,gt_kernels):
    
    total_tag = []
    total_label_kernel = []
    total_label_text = []
    for batch_i in range(batch):
        tag, label_kernel,label_text = get_cat_tag(outputs[batch_i, :].unsqueeze(0),gt_texts[batch_i],gt_kernels[batch_i])
        total_tag.append(tag)
        total_label_kernel.append(label_kernel)
        total_label_text.append(label_text)
    return total_tag,total_label_kernel,total_label_text
        
def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))
    
    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask
    
    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))
    
    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
    return selected_mask

def ohem_batch(scores, gt_texts, training_masks):
    scores = scores.data.cpu().numpy()
    gt_texts = gt_texts.data.cpu().numpy()
    training_masks = training_masks.data.cpu().numpy()

    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = np.concatenate(selected_masks, 0)
    selected_masks = torch.from_numpy(selected_masks).float()

    return selected_masks

def dice_loss(input, target, mask):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    mask = mask.contiguous().view(mask.size()[0], -1)
    
    input = input * mask
    target = target * mask

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = torch.sigmoid(texts).data.cpu().numpy() * training_masks
    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text >  0.5] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text

def cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel):
    mask = (gt_texts * training_masks).data.cpu().numpy()
    kernel = kernels#[:, -1, :, :]
    gt_kernel = gt_kernels#[:, -1, :, :]
    pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
    pred_kernel[pred_kernel <= 0.5] = 0
    pred_kernel[pred_kernel >  0.5] = 1
    pred_kernel = (pred_kernel * mask).astype(np.int32)
    gt_kernel = gt_kernel.data.cpu().numpy()
    gt_kernel = (gt_kernel * mask).astype(np.int32)
    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel

def get_index_t(tag,temp,label_kernel,index_t):
    a = np.array(tag)
    c = a[:, 1]
    b = list(set(c.tolist()))
    get_ = []
    for item in b:
        if (len(a[c == item]) > 1):
            get_.append(item)
    if (temp in get_):
        out = a[c == temp][:, 0]
        out = out[out != temp]
        for item in out:
            index_t = (label_kernel!= item) * index_t
    return index_t

def get_tag_gt(item):
    tag = []
    for i in range(1,int(item.max())+1):
        if((item==i).sum()>0):
            tag.append(i) 
    return tag

def get_kernel_compose(tag):
    get_i = 0
    out =[]
    while(get_i<(len(tag)-1)):
        for get_j in range(get_i+1,len(tag)):
            out.append([tag[get_i],tag[get_j]])
            out.append([tag[get_j],tag[get_i]])
        get_i+=1
    return out

def cal_Ldis_single_gt(similarity_vector, gt_compose, gt_kernel):
    
    lgg = 3
    loss_sum = []
#     print('gt_compose',gt_compose)
    for tag_s in gt_compose:
        index_k_i = (gt_kernel == tag_s[0]) 
        similarity_vector_k_i = torch.sum(similarity_vector[index_k_i], 0) / similarity_vector[index_k_i].shape[0]
        index_k_j= (gt_kernel == tag_s[1])
        similarity_vector_k_j = torch.sum(similarity_vector[index_k_j], 0) / similarity_vector[index_k_j].shape[0]
        out = torch.max(lgg-torch.norm(similarity_vector_k_i - similarity_vector_k_j),torch.tensor(0).float().cuda()).pow(2)
        out = torch.log(out+1) 
        loss_sum.append(out)
    if(len(loss_sum)==0):
        loss_sum = torch.tensor(0).float().cuda()
    else:
        loss_sum = torch.mean(torch.stack(loss_sum).cuda())
    return loss_sum

def cal_Ldis_gt(similarity_vector,gt_kernels_key,training_mask):
    similarity_vector = similarity_vector.permute((0, 2, 3, 1))
    Ldis_loss = []
    batch = similarity_vector.shape[0]
    for i in range(batch):
        tag = get_tag_gt(gt_kernels_key[i]*training_mask[i])
        if(len(tag)<2):
            continue
        gt_compose = get_kernel_compose(tag)
        loss_single = cal_Ldis_single_gt(similarity_vector[i], gt_compose, gt_kernels_key[i])
        Ldis_loss.append(loss_single)
        
    if(len(Ldis_loss)==0):
        Ldis_loss = torch.tensor(0.0).cuda()
    else:
        Ldis_loss = torch.mean(torch.stack(Ldis_loss).cuda())
    return Ldis_loss

def cal_Lagg_single_gt(similarity_vector,tag_kernel,gt_text,gt_kernel):
    agg = 0.5
    sum_agg = []
    for item in tag_kernel:
        index_k = (gt_kernel==item)
        index_t = (gt_text==item)
        similarity_vector_t = similarity_vector[index_t]
        similarity_vector_k =torch.sum(similarity_vector[index_k],0)/similarity_vector[index_k].shape[0]
        out = torch.norm((similarity_vector_t-similarity_vector_k),2,1) - agg
        out = torch.max(out, torch.tensor(0).float().cuda()).pow(2)
        ev_ =torch.log(out+1).mean()
        sum_agg.append(ev_)
    if(len(sum_agg)==0):
        loss_single = torch.tensor(0.0).cuda()
    else:
        loss_single = torch.mean(torch.stack(sum_agg).cuda())
    return loss_single

def cal_Lagg_gt(similarity_vector,gt_kernels_key,gt_text_key,training_mask):
    similarity_vector = similarity_vector.permute((0, 2, 3, 1))
    Lagg_loss = []
    batch = similarity_vector.shape[0]
    for i in range(batch):
        tag_kernel = get_tag_gt(gt_kernels_key[i]*training_mask[i])
        tag_text = get_tag_gt(gt_text_key[i]*training_mask[i])
        if(len(tag_kernel)<1 or len(tag_kernel)!= len(tag_text)):
            continue
        loss_single = cal_Lagg_single_gt(similarity_vector[i],tag_kernel,gt_text_key[i],gt_kernels_key[i])
        Lagg_loss.append( loss_single)
    if(len(Lagg_loss)==0):
        Lagg_loss = torch.tensor(0.0).cuda()
    else:
        Lagg_loss =torch.mean(torch.stack(Lagg_loss).cuda())
    return Lagg_loss

def train(train_loader, model, criterion, optimizer, epoch,writer):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_Lagg = AverageMeter()
    losses_Ldis = AverageMeter()
    running_metric_text = runningScore(2)
    running_metric_kernel = runningScore(2)

    end = time.time()
    
    for batch_idx, (imgs, gt_texts, gt_kernels, training_masks,gt_text_key,gt_kernels_key) in enumerate(train_loader):
        data_time.update(time.time() - end)
#         print(imgs.shape,gt_texts.shape,gt_kernels.shape,training_masks.shape,gt_text_key.shape,gt_kernels_key.shape)
        
        imgs = Variable(imgs.cuda()) # batch_size*channel*w*h
        gt_texts = Variable(gt_texts.cuda())# batch_size*w*h
        gt_kernels = Variable(gt_kernels.cuda())# batch_size*1*w*h
        gt_text_key = Variable(gt_text_key.cuda())# batch_size*w*h
        gt_kernels_key = Variable(gt_kernels_key.cuda())# batch_size*w*h
        training_masks = Variable(training_masks.cuda())# batch_size*w*h

        outputs = model(imgs)
        
        ind = 'cat_34'
        cv2.imwrite('text'+str(ind)+'.jpg',torch.sigmoid(outputs[0, 0, :, :]).data.cpu().numpy().astype(np.uint8)*255)
        cv2.imwrite('kernel'+str(ind)+'.jpg',torch.sigmoid(outputs[0, 1, :, :]).data.cpu().numpy().astype(np.uint8)*255)
        if batch_idx % 20 == 0:
            writer.add_image('/data/ori_image', torchvision.utils.make_grid(imgs,nrow=8, padding=10,normalize=True).cpu(),0)
            writer.add_image('/data/label_text', torchvision.utils.make_grid(gt_texts.cpu().unsqueeze(1),nrow=8, padding=10,normalize=True),0)
            writer.add_image('/data/predict_text', torchvision.utils.make_grid(torch.sigmoid(outputs[:, 0, :, :]).unsqueeze(1),nrow=8, padding=10,normalize=True).cpu(),0)
            writer.add_image('/data/label_kernel', torchvision.utils.make_grid(gt_kernels,nrow=8, padding=10,normalize=True).cpu(),0)
            writer.add_image('/data/predict_kernel', torchvision.utils.make_grid(torch.sigmoid(outputs[:, 1, :, :]).unsqueeze(1),nrow=8, padding=10,normalize=True).cpu(),0)
        
        
        texts = outputs[:, 0, :, :]
        kernels = outputs[:, 1:2, :, :]

        similarity_vector=outputs[:,2:,:,:]#torch.sigmoid(outputs[:,2:,:,:])

        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        selected_masks = Variable(selected_masks.cuda())

        loss_text = criterion(texts, gt_texts, selected_masks)
        

        mask0 = torch.sigmoid(texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = Variable(selected_masks.cuda())

        loss_kernels = []
        for i in range(1):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = criterion(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)
        loss_kernel = sum(loss_kernels) / len(loss_kernels)

        
#         total_tag,total_label_kernel,total_label_text = get_batch_tag(outputs,outputs.shape[0],gt_texts,gt_kernels)
#         loss_agg = cal_Lagg(similarity_vector,total_tag,total_label_kernel,total_label_text)
#         loss_ldis = cal_Ldis(similarity_vector,total_tag,total_label_kernel)

        loss_agg = cal_Lagg_gt(similarity_vector,gt_kernels_key,gt_text_key,training_masks)
        loss_ldis = cal_Ldis_gt(similarity_vector,gt_kernels_key,training_masks)

#         loss_agg,loss_ldis = agg_dis_loss(outputs[:, 0, :, :], outputs[:, 1, :, :], gt_text_key, gt_kernels_key, similarity_vector)
#         loss_agg = loss_agg.mean()
#         loss_ldis = loss_ldis.mean()
        
        loss = loss_text + 0.5*loss_kernel+0.25*(loss_agg+loss_ldis)
        
        writer.add_scalar('Loss/total_loss',loss,batch_idx+epoch*(1000/8))
        writer.add_scalar('Loss/loss_text',loss_text,batch_idx+epoch*(1000/8))
        writer.add_scalar('Loss/loss_kernel',loss_kernel,batch_idx+epoch*(1000/8))
        writer.add_scalar('Loss/loss_agg',loss_agg,batch_idx+epoch*(1000/8))
        writer.add_scalar('Loss/loss_ldis',loss_ldis,batch_idx+epoch*(1000/8))
        
        losses.update(loss.item(), imgs.size(0))
        losses_Lagg.update(loss_agg.item(), imgs.size(0))
        losses_Ldis.update(loss_ldis.item(), imgs.size(0))
        
        optimizer.zero_grad()
        loss.backward()
#         print('loss_text',loss_text.grad)
#         print('loss_kernel',loss_kernel.grad)
#         print('loss_agg',loss_agg.grad)
#         print('loss_ldis',loss_ldis.grad)
#         print('loss',loss.grad)
        optimizer.step()

        score_text = cal_text_score(texts, gt_texts, training_masks, running_metric_text)
        score_kernel = cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 20 == 0:
            output_log  = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min | Loss: {loss:.4f} | Acc_t: {acc: .4f} | IOU_t: {iou_t: .4f} | IOU_k: {iou_k: .4f} | Lagg: {lagg:.4f} | Ldis: {ldis:.4f}'.format(
                batch=batch_idx + 1,
                size=len(train_loader),
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                loss=losses.avg,
                acc=score_text['Mean Acc'],
                iou_t=score_text['Mean IoU'],
                iou_k=score_kernel['Mean IoU'],
                lagg=losses_Lagg.avg,
                ldis=losses_Ldis.avg
            )
            print(output_log)
            sys.stdout.flush()

    return (losses.avg, score_text['Mean Acc'], score_kernel['Mean Acc'], score_text['Mean IoU'], score_kernel['Mean IoU'],losses_Lagg.avg,losses_Ldis.avg)

def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def main(args):
    if args.checkpoint == '':
        args.checkpoint = "checkpoints/ic15_%s_bs_%d_ep_%d"%(args.arch, args.batch_size, args.n_epoch)
    if args.pretrain:
        if 'synth' in args.pretrain:
            args.checkpoint += "_pretrain_synth"
        else:
            args.checkpoint += "_pretrain_ic17"

    print ('checkpoint path: %s'%args.checkpoint)
    print ('init lr: %.8f'%args.lr)
    print ('schedule: ', args.schedule)
    sys.stdout.flush()

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    kernel_num = 2
    min_scale = 0.4
    start_epoch = 0

    data_loader = IC15Loader(is_transform=True, img_size=args.img_size, kernel_num=kernel_num, min_scale=min_scale)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=5,
        drop_last=True,
        pin_memory=True)

    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=kernel_num+4)
    elif args.arch == "resnet50_fxw":
        model = models.resnet50(pretrained=True, num_classes=kernel_num+4)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=kernel_num)
    
    model = torch.nn.DataParallel(model).cuda()
    
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)

    title = 'icdar2015'
    if args.pretrain:
        print('Using pretrained model.')
        assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.','Lagg Loss','Ldis Loss'])
    elif args.resume:
        print('Resuming from checkpoint.')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        print('Training from scratch.')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.','Lagg Loss','Ldis Loss'])
    
    from tensorboardX import SummaryWriter
    tag = 'remove_kernel_fxw'
    writer = SummaryWriter(log_dir='./logs_resnet50/'+tag)
    for epoch in range(start_epoch, args.n_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.n_epoch, optimizer.param_groups[0]['lr']))

        train_loss, train_te_acc, train_ke_acc, train_te_iou, train_ke_iou,losses_Lagg,losses_Ldis = train(train_loader, model, dice_loss, optimizer, epoch,writer)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lr': args.lr,
                'optimizer' : optimizer.state_dict(),
            }, checkpoint=args.checkpoint)

        logger.append([optimizer.param_groups[0]['lr'], train_loss, train_te_acc, train_te_iou,losses_Lagg,losses_Ldis])
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50_fxw')
    parser.add_argument('--img_size', nargs='?', type=int, default=640, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=600, 
                        help='# of the epochs')
    parser.add_argument('--schedule', type=int, nargs='+', default=[10,20,30, 40,50,60,70,80,90,100,150,200,250,300,350,400,450],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--pretrain', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    args = parser.parse_args()

    main(args)

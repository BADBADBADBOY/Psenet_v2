import os
os.environ["CUDA_VISIBLE_DEVICES"] ="1"
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data
import time
from dataset import IC15TestLoader
import models
import util

def write_result_as_txt(image_name, bboxes, path):
    filename = util.io.join_path(path, 'res_%s.txt'%(image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)


def get_label_num(text,min_area):
    label_num, label = cv2.connectedComponents(text, connectivity=4)
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            label_num-=1
    return label_num,label

def get_cat_tag(text,kernel):
    

    text = text.data.cpu().numpy().astype(np.uint8)
    kernel = kernel.data.cpu().numpy().astype(np.uint8)
   
    label_num_text, label_text = get_label_num(text, 100)
    label_num_kernel, label_kernel = get_label_num(kernel, 100)

    tag = []
    for i in range(1, int(label_kernel.max()) + 1):
        if ((label_kernel == i).sum() > 0):
            tag.append(i)
    tag_cat = []
    for i in tag:
        tag_cat.append([i, ((label_kernel == i) * label_text).max()])
    return tag_cat,label_kernel,label_text

def get_kernel_compose(tag):
    get_i = 0
    out =[]
    while(get_i<(len(tag)-1)):
        for get_j in range(get_i+1,len(tag)):
            out.append([tag[get_i],tag[get_j]])
        get_i+=1
    return out

def scale_long(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def scale_short(img, short_size=736):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def test(args):
    data_loader = IC15TestLoader(long_size=args.long_size)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=6, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=7, scale=args.scale)
    
    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda()
    
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            
            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.stdout.flush()

    model.eval()
    
    total_frame = 0.0
    total_time = 0.0
    bboxs = []
    bboxes = []
    for idx, (org_img, img) in enumerate(test_loader):
        print('progress: %d / %d'%(idx, len(test_loader)))
        sys.stdout.flush()

        img = Variable(img.cuda())
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

        
        
        with torch.no_grad():
            outputs = model(img)
            
        torch.cuda.synchronize()
        start = time.time()
        
        similarity_vector=outputs[0,2:,:,:]
        similarity_vector_ori = similarity_vector.permute(( 1, 2, 0))
        
        score = torch.sigmoid(outputs[:, 0, :, :])
        score = score.data.cpu().numpy()[0].astype(np.float32)
        
        outputs = (torch.sign(outputs - 1.0) + 1) / 2
    
        text = outputs[0, 0, :, :]
        kernel =outputs[0, 1, :, :] * text
        
       
        

        tag_cat,label_kernel,label_text = get_cat_tag(text,kernel)
        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
#         cv2.imwrite('./test_result/image/text_'+image_name+'.jpg',label_text*255)
#         cv2.imwrite('./test_result/image/kernel_'+image_name+'.jpg',label_kernel*255)
        
        label_text = torch.Tensor(label_text).cuda()
        label_kernel = torch.Tensor(label_kernel).cuda()
        
        w,h ,_= similarity_vector_ori.shape
        similarity_vector = similarity_vector.permute(( 1, 2, 0)).data.cpu().numpy()
        bboxs = []
        bboxes = []
        scale = (org_img.shape[1] * 1.0 / text.shape[1], org_img.shape[0] * 1.0 / text.shape[0])
        
        for item in tag_cat: 
            
            similarity_vector_ori1 = similarity_vector_ori.clone()
#             mask = torch.zeros((w,h)).cuda()
            index_k = (label_kernel==item[0])
            index_t = (label_text==item[1])
            similarity_vector_k =torch.sum(similarity_vector_ori1[index_k],0)/similarity_vector_ori1[index_k].shape[0]
#             similarity_vector_t = similarity_vector_ori1[index_t]
            
#             similarity_vector_t = similarity_vector_ori1[index_t]
            similarity_vector_ori1[~index_t] = similarity_vector_k
            similarity_vector_ori1 = similarity_vector_ori1.reshape(-1,4)
            out = torch.norm((similarity_vector_ori1-similarity_vector_k),2,1)

#             out = torch.norm((similarity_vector_t-similarity_vector_k),2,1)
#             print(out.shape)
#             mask[index_t] = out
            out = out.reshape(w,h)
           
            out = out*((text>0).float())
#             out = mask*((text>0).float())
    
            out[out>0.8]=0
            out[out>0]=1
            out_im = (text*out).data.cpu().numpy()
#             cv2.imwrite('./test_result/image/out_'+image_name+'.jpg',out_im*255)
            points = np.array(np.where(out_im == out_im.max())).transpose((1, 0))[:, ::-1]
            

            if points.shape[0] < 800 :
                continue

            score_i = np.mean(score[out_im == out_im.max()])
            if score_i < 0.93:
                continue

            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect) * scale
            bbox = bbox.astype('int32')
            bboxs.append(bbox)
            bboxes.append(bbox.reshape(-1))
            
#         text_box = scale(text_box, long_size=2240)
        torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print('fps: %.2f'%(total_frame / total_time))
        

        for bbox in bboxs:
            text_box =cv2.line(text_box, (bbox[0,0],bbox[0,1]),(bbox[1,0],bbox[1,1]), (0, 0, 255), 2) 
            text_box =cv2.line(text_box, (bbox[1,0],bbox[1,1]),(bbox[2,0],bbox[2,1]), (0, 0, 255), 2) 
            text_box =cv2.line(text_box, (bbox[2,0],bbox[2,1]),(bbox[3,0],bbox[3,1]), (0, 0, 255), 2) 
            text_box =cv2.line(text_box, (bbox[3,0],bbox[3,1]),(bbox[0,0],bbox[0,1]), (0, 0, 255), 2) 
        write_result_as_txt(image_name, bboxes, 'test_result/submit_ic15/')
        cv2.imwrite('./test_result/image/'+image_name+'.jpg',text_box)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--resume', nargs='?', type=str, default='./checkpoints/ic15_resnet50_test_test_test_bs_8_ep_600/checkpoint.pth.tar',    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=2,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=2240,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=800.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
                        help='min score')
    
    args = parser.parse_args()
    test(args)

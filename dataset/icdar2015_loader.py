# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg

ic15_root_dir = '/src/notebooks/train_data/'
ic15_train_data_dir = ic15_root_dir + 'ch4_training_images/'
ic15_train_gt_dir = ic15_root_dir + 'ch4_training_gts/'
ic15_test_data_dir = ic15_root_dir + 'ch4_training_images/'
ic15_test_gt_dir = ic15_root_dir + 'ch4_training_gts/'
random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print (img_path)
        raise
    return img

def get_bboxes(img, gt_path):
    h, w = img.shape[0:2]
    # lines = util.io.read_lines(gt_path)
    with open(gt_path,'r',encoding='utf-8') as fid:
        lines = fid.readlines()
    bboxes = []
    tags = []
    for line in lines:
        line = line.replace('\ufeff','')
        line = util.str.remove_all(line, '\xef\xbb\xbf')
        gt = util.str.split(line, ',')
        if gt[-1][0] == '#':
            tags.append(False)
        else:
            tags.append(True)
        box = [int(gt[i]) for i in range(8)]
        box = np.asarray(box) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(box)
    return np.array(bboxes), tags

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def cal_affine_coord(ori_coord,M):
    x = ori_coord[0]
    y = ori_coord[1]
    _x = x * M[0, 0] + y * M[0, 1] + M[0, 2]
    _y = x * M[1, 0] + y * M[1, 1] + M[1, 2]
    return [int(_x),int(_y)]

def get_rotate_bboxs(bboxs,M):
    out_bboxs=[]
    for bbox in bboxs:
        box = []
        for i in range(4):
           out = cal_affine_coord(bbox[i,:],M)
           box.append(out)
        out_bboxs.append(box)
    return np.array(out_bboxs)

def random_rotate(img):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    w, h = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
    return img_rotation,rotation_matrix

def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_scale(img, min_size):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[0:2]
    random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs
    
    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis = 1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis = 1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        
        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
    
    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bbox = np.array(shrinked_bbox)[0]
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bboxes.append(shrinked_bbox)
    
    return np.array(shrinked_bboxes)

class IC15Loader(data.Dataset):
    def __init__(self, is_transform=False, img_size=None, kernel_num=7, min_scale=0.4):
        self.is_transform = is_transform
        
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_num = kernel_num
        self.min_scale = min_scale

        data_dirs = [ic15_train_data_dir]
        gt_dirs = [ic15_train_gt_dir]

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))
            # img_names.extend(util.io.ls(data_dir, '.gif'))

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)
                
                gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        bboxes, tags = get_bboxes(img, gt_path)
        
        if self.is_transform:
            cv2.imwrite('ori.jpg',img)
            img = random_scale(img, self.img_size[0])
            cv2.imwrite('scale.jpg',img)
            img,rotation_matrix = random_rotate(img)
            cv2.imwrite('rotate.jpg',img)
        gt_text = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4), (bboxes.shape[0], bboxes.shape[1] // 2, 2)).astype('int32')
            bboxes = get_rotate_bboxs(bboxes,rotation_matrix)
            img_ori = img.copy()
            for item in bboxes:
                img_ori = cv2.line(img_ori,(item[0,0],item[0,1]),(item[1,0],item[1,1]),(0,0,255))
                img_ori = cv2.line(img_ori,(item[1,0],item[1,1]),(item[2,0],item[2,1]),(0,0,255))
                img_ori = cv2.line(img_ori,(item[2,0],item[2,1]),(item[3,0],item[3,1]),(0,0,255))
                img_ori = cv2.line(img_ori,(item[3,0],item[3,1]),(item[0,0],item[0,1]),(0,0,255))
            cv2.imwrite('show.jpg',img_ori)
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
                if not tags[i]:
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for i in range(1, self.kernel_num):
#             rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            rate = 0.5
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for j in range(bboxes.shape[0]):
                cv2.drawContours(gt_kernel, [kernel_bboxes[j]], -1, j+1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_text, training_mask]
            imgs.extend(gt_kernels)

            imgs = random_horizontal_flip(imgs)
#             imgs = random_rotate(imgs)
            imgs = random_crop(imgs, self.img_size)

            img, gt_text, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]
        

        gt_kernels = np.array(gt_kernels)

        cv2.imwrite('kernel.jpg',gt_kernels[0]*40)
        cv2.imwrite('text.jpg',gt_text*40)
        cv2.imwrite('training_mask.jpg',training_mask*255)
        
        gt_kernels_key = gt_kernels[0].copy()
        gt_text_key = gt_text.copy()
        
#         gt_text[gt_text > 0] = 1
#         gt_kernels[0][gt_kernels[0] > 0] = 1
        
#         tag_text = []
#         tag_kernel = []
#         for i in range(1,gt_kernels_key.max()+1):
#             if((gt_kernels_key==i).sum()>0):
#                 tag_kernel.append(i)
#         for i in range(1,gt_text_key.max()+1):
#             if((gt_text_key==i).sum()>0):
#                 tag_text.append(i)
#         print('tag_text',tag_text)
#         print('tag_kernel',tag_kernel)
        
#         if(len(tag_text)!=len(tag_kernel)):
#             print(bbb)
        # '''
        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness = 32.0 / 255, saturation = 0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).float()
        gt_kernels = torch.from_numpy(gt_kernels).float()
        training_mask = torch.from_numpy(training_mask).float()
        gt_text_key = torch.from_numpy(gt_text_key).float()
        gt_kernels_key = torch.from_numpy(gt_kernels_key).float()
        # '''

        return img, gt_text, gt_kernels, training_mask,gt_text_key,gt_kernels_key
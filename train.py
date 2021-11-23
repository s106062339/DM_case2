import os
import cv2
import random
# import pydicom
import argparse
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from TransFG.modeling import VisionTransformer
import TransFG.configs as configs

torch.manual_seed(0)

def print_basic_info(args_info):
    
    print('Using PyTorch version', torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print('Default GPU Device: {}'.format(torch.cuda.get_device_name(0)))
    else:
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')

    print(args_info)

###############################################################################################
################################### hyperparameter settings ###################################
###############################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--kfold', type=int, default=0)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--resampling', type=int, default=2)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--image_size', type=int, default=528)
parser.add_argument('--root_dir', type=str, default="/home/cjho/DM_CASE2")
parser.add_argument('--saved_model_path', type=str, default="/home/cjho/DM_CASE2/save_model/ViT_1124.pth")
args = parser.parse_args()
print_basic_info(args)



######################################################################################
################################### model settings ###################################
######################################################################################

config = configs.get_b16_config()
# config = configs.get_l32_config()
config.split = 'non-overlap'
model = VisionTransformer(config, args.image_size, num_classes=args.num_classes)
model_weight_path = os.path.join(args.root_dir, "pretrain_weight", "ViT-B_16.npz")
model.load_from(np.load(model_weight_path))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience, verbose=True)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

######################################################################################
#################################### data settings ###################################
######################################################################################

class hisEqulColor(object):
    def __call__(self, img):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB, img)
        return Image.fromarray(img)
    def __repr__(self):
        return self.__class__.__name__+'()'

class CLAHE(object):
    def __call__(self, img):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)
    def __repr__(self):
        return self.__class__.__name__+'()'
        
class ChestXRayDataset(torch.utils.data.Dataset):
    
    def __init__(self, transform , mode):
        
        self.mode = mode
        self.transform  = transform 
        self.class2index = {'normal' : 0, 'pneu' : 1, 'covid' : 2}
        
        self.data_indice = os.path.join(args.root_dir, "dcm_folds/" + mode + "_fold" + str(args.kfold) + ".txt")
        self.data_images = []

        with open(self.data_indice, "r") as f:
            content = f.readlines()
            for item in content:
                image_name, class_name = item[:-1].split(",")
                self.data_images.append([image_name[:-3] + "png", class_name])

            print("Found " + str(len(self.data_images)) + " " + mode + " examples.")

        if mode == "train":
            self.data_images = self.data_images * args.resampling
            print("After resmapling : " + str(len(self.data_images)) + " " + mode + " examples.")
            
        random.shuffle(self.data_images)

    def __len__(self):
        return len(self.data_images)
    
    
    def __getitem__(self, index):
        
        img_name, label = self.data_images[index]
    
        img_path = os.path.join(args.root_dir, "train_data", label + "s", img_name)
        
        image = self.transform(Image.fromarray(cv2.imread(img_path)))

        return image, self.class2index[label]


train_transform = transforms.Compose([
        transforms.Resize((int(args.image_size * 1.15), int(args.image_size * 1.15))),
        transforms.RandomAffine(degrees=20, scale=(0.85, 1.15)),
        transforms.RandomCrop((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        hisEqulColor(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

val_transform = transforms.Compose([
        transforms.Resize((int(args.image_size * 1.15), int(args.image_size * 1.15))),
        transforms.CenterCrop((args.image_size, args.image_size)),
        hisEqulColor(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = ChestXRayDataset(train_transform, "train")
valid_dataset = ChestXRayDataset(val_transform,   "valid")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

############################################################################################
#################################### training processing ###################################
############################################################################################

def train(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    train_correct_count = 0
    count = 0

    gt_list   = []
    pred_list = []

    with tqdm(enumerate(train_loader), total = len(train_loader)) as pbar:
        for i, data in pbar:
            imgs, labels = data
            imgs = imgs.to(device)        
            labels = labels.to(device)  
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            train_correct_count += (preds == labels).sum().item()
            
            for gt_item, pred_item in zip(labels.tolist(), preds.tolist()):
                gt_list.append(gt_item)
                pred_list.append(pred_item)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            count += 1
    
    epoch_loss = train_loss / count
    epoch_acc = train_correct_count / len(train_loader.dataset)
    epoch_f1score = f1_score(gt_list, pred_list, average='macro')

    return epoch_loss, epoch_acc, epoch_f1score

def val(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    val_correct_count = 0
    count = 0
    gt_list   = []
    pred_list = []

    with torch.no_grad():
        with tqdm(enumerate(val_loader), total = len(val_loader)) as pbar:
            for i, data in pbar:
                imgs, labels = data
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                val_correct_count += (preds == labels).sum().item()
                
                for gt_item, pred_item in zip(labels.tolist(), preds.tolist()):
                    gt_list.append(gt_item)
                    pred_list.append(pred_item)
                    
                count += 1
    
    epoch_loss = val_loss / count
    epoch_acc = val_correct_count / len(val_loader.dataset)
    epoch_f1score = f1_score(gt_list, pred_list, average='macro')

    return epoch_loss, epoch_acc, epoch_f1score

max_val_f1score = 0

for e in range(args.epoch):
    
    print("epoch =", (e + 1))

    train_loss, train_acc, train_f1score = train(model, train_loader, optimizer, criterion)
    print("train loss =", round(train_loss, 4), "train_acc =", round(train_acc, 4), "train_f1score =", round(train_f1score, 4))
    
    val_loss, val_acc, val_f1score = val(model, val_loader, criterion)
    print("val loss   =", round(val_loss, 4), "val_acc   =", round(val_acc, 4), "val_f1score   =", round(val_f1score, 4), "\n")

    scheduler.step(val_loss)
    
    if val_f1score >= max_val_f1score:
        max_val_f1score = val_f1score
        torch.save(model.state_dict(), args.saved_model_path[:-4] + "_" + str(e+1) + "_" + str(round(val_f1score, 4)) + ".pth")

    elif (e + 1) % 5 == 0 or val_f1score > 0.6:
        torch.save(model.state_dict(), args.saved_model_path[:-4] + "_" + str(e+1) + "_" + str(round(val_f1score, 4)) + ".pth")
    
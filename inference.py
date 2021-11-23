import os
import cv2
import csv
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

import torch
from torchvision import transforms

from TransFG.modeling import VisionTransformer
import TransFG.configs as configs


parser = argparse.ArgumentParser()
parser.add_argument('--pthfile', type=str, default="")
parser.add_argument('--image_size', type=int, default=528)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--root_dir', type=str, default="/home/cjho/DM_CASE2")
args = parser.parse_args()

pthfile = os.path.join(args.root_dir, "save_model", args.pthfile)

print(pthfile)

def build_b16(model_path):
    config = configs.get_b16_config()
    config.split = 'non-overlap'
    model = VisionTransformer(config, args.image_size, num_classes=args.num_classes)

    state_dict = torch.load(model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model

model = build_b16(pthfile)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class ChestXRayDataset(torch.utils.data.Dataset):
    
    def __init__(self, transform):
        
        self.transform = transform

        self.class2index = {'normal' : 0, 'pneu' : 1, 'covid' : 2}

        self.data_images = []
        for item in os.listdir( os.path.join(args.root_dir, "test_data")):
            if item[-3:] == "png":
                self.data_images.append(item)

    def __len__(self):
        return len(self.data_images)
    
    
    def __getitem__(self, index):
        
        img_path = os.path.join(args.root_dir, "test_data", self.data_images[index])

        data = cv2.imread(img_path)
        image = self.transform(Image.fromarray(data))

        return image, img_path

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

val_transform = transforms.Compose([
        transforms.Resize((int(args.image_size * 1.15), int(args.image_size * 1.15))),
        transforms.CenterCrop((args.image_size, args.image_size)),
        hisEqulColor(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

valid_dataset = ChestXRayDataset(val_transform)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

model.eval()

class2index = {'normal' : 0, 'pneu' : 1, 'covid' : 2}

id2name = {0 : "Negative", 1 : "Typical", 2 : "Atypical"}

with open('./output_submission.csv', "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['FileID', 'Type'])

    with tqdm(enumerate(valid_loader), total = len(valid_loader)) as pbar:
        for i, data in pbar:
            
            imgs, img_path = data
            imgs = imgs.to(device)
            
            output = model(imgs)

            _, preds = torch.max(output.data, 1)
            
            for fileid, type in zip(img_path, preds.tolist()):
                writer.writerow([os.path.basename(fileid)[:-4], id2name[type]])
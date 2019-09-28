import os
import sys
import scipy.io as sio
import numpy as np
import cv2 as cv

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset

from PIL import Image

from deep.siamese import BranchNetwork, SiameseNetwork


#################### data processing ####################
# calculate transforms.Normalize(normMean = {}, normStd = {})
def cal_mean_std(img_h, img_w, num, root):
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []

    img_list = os.listdir(root)
    img_list = [os.path.join(root, k) for k in img_list]

    for i in range(num):
        img = cv.imread(img_list[i])
        img = cv.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]

        imgs = np.concatenate((imgs, img), axis=3)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

    return means[0], stdevs[0]


# resize edge images to (1280 × 720)->(w, h)
size_transform = transforms.Compose(
    [transforms.Resize([720, 1280]),
     transforms.ToTensor()])

# transform = transforms.Compose(
#     [transforms.Resize([180, 320]),
#         transforms.ToTensor()])

# transform = transforms.Compose(
#     [transforms.Resize([180, 320]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.0188], std=[0.128])])

edge_image_dir = "../data/worldcup2014/edge_images/"
img_h = 180
img_w = 320
num = 186  # the number of all the test images

mean, std = cal_mean_std(img_h, img_w, num, edge_image_dir)
print('worldcup2014 dataset mean: {}'.format(mean))
print('worldcup2014 dataset std: {}'.format(std))

# mean = 0.0218
# std = 0.115
transform = transforms.Compose(
    [transforms.Resize([180, 320]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])])


class imageset(Dataset):
    def __init__(self, root):
        # root: the path to edge images
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path).convert('L')
        data = self.transforms(pil_img)
        data = data.unsqueeze(1)
        return data

    def __len__(self):
        return len(self.imgs)


# create edge_map just like 'testset_feature.mat'
edge_map = []
edge_image_list = os.listdir(edge_image_dir)
edge_image_list = sorted(edge_image_list, key=lambda x: int(x.split('_')[0]))

for i in edge_image_list:
    img_path = os.path.join(edge_image_dir, i)
    im = Image.open(img_path).convert('L')
    im = size_transform(im).squeeze(0).numpy()
    im = np.transpose((im))
    edge_map.append(im)


edge_map = torch.Tensor(edge_map)
edge_map = np.transpose(edge_map).unsqueeze(2).numpy()

edge_image_dataset = imageset(edge_image_dir)

#################### extract features ####################
# this part imitate the 'network_test.py'

# load network
branch = BranchNetwork()
siamese_network = SiameseNetwork(branch)

# setup computation device
device = 'cpu'
cuda_id = 0
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(cuda_id))
    siamese_network = siamese_network.to(device)
    cudnn.benchmark = True
print('computation device: {}'.format(device))

model_name = 'network.pth'
current_dir = os.getcwd()
model_dir = os.path.join(current_dir, 'deep')

if model_name in os.listdir(model_dir):
    model_dir = os.path.join(model_dir, model_name)
    checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage)
    siamese_network.load_state_dict(checkpoint['state_dict'])
    print('load model file from {}.'.format(model_name))
else:
    print('Error: file not found at {}'.format(model_name))
    sys.exit()

features = []
with torch.no_grad():
    for i in range(edge_image_dataset.__len__()):
        x = edge_image_dataset[i]
        x = x.to(device)
        feat = siamese_network.feature_numpy(x)  # N x C

        features.append(feat)

        print('finished {} in {}'.format(i + 1, edge_image_dataset.__len__()))

features = np.vstack((features))
features = np.transpose(features)
print('feature dimension {}'.format(features.shape))

save_file = "../data/worldcup2014/features_result/testset_feature_new.mat"
# sio.savemat(save_file, {'features_new': features,
#                         'edge_map_new': edge_map})
sio.savemat(save_file, {'features_new': features})
print('save to {}'.format(save_file))

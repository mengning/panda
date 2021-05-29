import os
import cv2
import skimage.io
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import model as enet
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

data_dir = 'D:\\working\\kaggle\\data'
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
image_dir = os.path.join(data_dir, 'train_images')
model_dir = 'D:\\working\\kaggle\\panda-public-models'

input_size = 1536
tile_size = 128
n_tiles = (input_size // tile_size) ** 2
batch_size = 2
height = 0
weight = 0
df = df[5:9]
device = torch.device('cuda')


class MyEfficient(nn.Module):
    def __init__(self, backbone, out_dim):
        super(MyEfficient, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


def load_models(model_files):
    models = []
    for model_f in model_files:
        model_f = os.path.join(model_dir, model_f)
        backbone = 'efficientnet-b0'
        model = MyEfficient(backbone, out_dim=5)
        model.load_state_dict(torch.load(model_f, map_location=lambda storage, loc: storage), strict=True)
        model.eval()
        model.to(device)
        models.append(model)
        print(f'{model_f} loaded!')
    return models


model_files = [
    'cls_effnet_b0_Rand36r36tiles256_big_bce_lr0.3_augx2_30epo_model_fold0.pth'
]

models = load_models(model_files)


def get_tiles(img, index):
    result = []
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    # w_len是按tile_size拆分后宽度方面小块的个数
    w_len = (w + pad_w) // tile_size
    # row_n,col_n分别是第index个小方块的横坐标和纵坐标
    row_n = index // w_len
    col_n = index % w_len
    # row,col分别是第index个小方块的左上角像素位置
    row = row_n * tile_size
    col = col_n * tile_size
    img = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]], constant_values=255)
    img = img[row:row + tile_size, col:col + tile_size, :]
    for i in range(n_tiles):
        result.append({'img': img, 'idx': i})
    return result


def get_size(img):
    h, w, c = img.shape
    print('原始:h ={}, w = {}'.format(h, w))
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size

    h = (h + pad_h) // tile_size
    w = (w + pad_w) // tile_size
    print('填充后:h ={}, w = {}'.format(h, w))
    return h, w


def get_num(img):
    h, w = get_size(img)
    return h * w


class PANDADataset(Dataset):
    def __init__(self, df, tile_size, n_tiles=n_tiles):

        self.df = df.reset_index(drop=True)
        self.tile_size = tile_size
        self.n_tiles = n_tiles

    def __len__(self):
        row = self.df.iloc[0]
        img_id = row.image_id

        tiff_file = os.path.join(image_dir, f'{img_id}.tiff')
        print("image name: ", tiff_file)
        image = skimage.io.MultiImage(tiff_file)[1]
        num = get_num(image)
        print("num = ", num)
        return num

    def __getitem__(self, index):
        row = self.df.iloc[0]
        img_id = row.image_id

        tiff_file = os.path.join(image_dir, f'{img_id}.tiff')
        image = skimage.io.MultiImage(tiff_file)[1]
        tiles = get_tiles(image, index)

        idxes = list(range(self.n_tiles))

        n_row_tiles = int(np.sqrt(self.n_tiles))
        images = np.zeros((self.tile_size * n_row_tiles, self.tile_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w

                if len(tiles) > idxes[i]:
                    this_img = tiles[idxes[i]]['img']
                else:
                    this_img = np.ones((self.tile_size, self.tile_size, 3)).astype(np.uint8) * 255
                this_img = 255 - this_img
                h1 = h * self.tile_size
                w1 = w * self.tile_size
                images[h1:h1 + self.tile_size, w1:w1 + self.tile_size, :] = this_img

        #         images = 255 - images
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        return torch.tensor(images)


def label_image(level, img, row, col):
    color_level = [[0, 100, 0], [0, 255, 0], [255, 255, 0], [0, 0, 255], [0, 0, 100]]
    # 标注top
    img[row, col:col + tile_size - 1] = color_level[int(level) - 1]
    # 标注down
    img[row + tile_size - 1, col + 1:col + tile_size] = color_level[int(level) - 1]
    # 标注left
    img[row + 1:row + tile_size, col] = color_level[int(level) - 1]
    # 标注right
    img[row:row + tile_size - 1, col + tile_size - 1] = color_level[int(level) - 1]
    cv2.putText(img, str(int(level)), (col, row + tile_size - 1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)


# def block(img, top, down, left, right):
#     if down - top <= 1080 and right - left <= 1920:
#         cv2.rectangle(img, (left, top), (left + 1920, top + 1080), (0, 255, 0), 2)
#     elif down - top <= 1080 and right - left > 1920:
#         width = right - left
#         while width > 1920:
#             cv2.rectangle(img, (left, top), (left + 1920, top + 1080), (0, 255, 0), 2)
#             width = width - 1920
#             left = left + 1800
#             width = width + 120
#
#         if 0 < width <= 1920:
#             cv2.rectangle(img, (left + width - 1920, top), (left + width, top + 1080), (0, 255, 0), 2)
#     elif down - top > 1080 and right - left <= 1920:
#         height = down - top
#         while height > 1080:
#             cv2.rectangle(img, (left, top), (left + 1920, top + 1080), (0, 255, 0), 2)
#             height = height - 1080
#             top = top + 900
#             height = height + 180
#
#         if 0 < height <= 1080:
#             cv2.rectangle(img, (left, top + height - 1080), (left + 1920, top + height), (0, 255, 0), 2)
#     else:
#         width = right - left
#         height = down - top
#         while height > 1080:
#             while width > 1920:
#                 cv2.rectangle(img, (left, top), (left + 1920, top + 1080), (0, 255, 0), 2)
#                 width = width - 1920
#                 left = left + 1800
#                 width = width + 120
#
#             if 0 < width <= 1920:
#                 cv2.rectangle(img, (left + width - 1920, top), (left + width, top + 1080), (0, 255, 0), 2)
#
#             height = height - 1080
#             top = top + 900
#             height = height + 180
#
#         if 0 < height <= 1080:
#             while width > 1920:
#                 cv2.rectangle(img, (left, top + height - 1080), (left + 1920, top + height), (0, 255, 0), 2)
#                 width = width - 1920
#                 left = left + 1800
#                 width = width + 120
#
#             if 0 < width <= 1920:
#                 cv2.rectangle(img, (left + width - 1920, top + height - 1080), (left + width, top + height), (0, 255, 0), 2)
#     return img
sigma = 100
width_display = 1920 - 100
height_display = 1080 - 100


# 以img的top,down,left,right确定的小块中的点集（点是预测等级大于0的方块中心点）坐标的均值为中心显示图像
def center_display(img, top, down, left, right, center_list):
    m = np.mean(center_list[(top <= center_list[:, 0]) & (center_list[:, 0] <= down) &
                            (left <= center_list[:, 1]) & (center_list[:, 1] <= right)], axis=0)
    left = int(m[1]) - (width_display + sigma) // 2
    right = int(m[1]) + (width_display + sigma) // 2
    top = int(m[0]) - (height_display + sigma) // 2
    down = int(m[0]) + (height_display + sigma) // 2
    cv2.rectangle(img, (left, top), (right, down), (255, 0, 0), 2)

    x = int(m[1])
    y = int(m[0])
    cv2.putText(img, 'center', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 255), 2)
    return img


# 重新计算top,down,left,right确定的小块中的有效高度（即小块中点集的最上界和最下界），按照有效高度拆成一个一个height_display来居中显示
def height_handle(img, top, down, left, right, center_list):
    new_top = np.min(center_list[(top <= center_list[:, 0]) & (center_list[:, 0] <= down) &
                            (left <= center_list[:, 1]) & (center_list[:, 1] <= right)], axis=0)[0] - tile_size // 2
    new_down = np.max(center_list[(top <= center_list[:, 0]) & (center_list[:, 0] <= down) &
                            (left <= center_list[:, 1]) & (center_list[:, 1] <= right)], axis=0)[0] + tile_size // 2
    new_height = new_down - new_top
    while new_height > height_display:
        img = center_display(img, new_top, new_top + height_display, left, right, center_list)
        new_top = new_top + height_display
        new_height = new_height - height_display
    return center_display(img, new_top, new_down, left, right, center_list)


# 将img的某一块按照屏幕物理分辨率来显示，按照宽度一块一块进行高度处理
def block(img, top, down, left, right, center_list):
    w = right - left
    h = down - top

    while w > width_display:
        img = height_handle(img, top, down, left, left + width_display, center_list)
        left = left + width_display
        w = w - width_display

    img = height_handle(img, top, down, left, right, center_list)
    return img


def gen_image(img):
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    w_len = (w + pad_w) // tile_size
    h_len = (h + pad_h) // tile_size
    img = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
                  constant_values=255)
    centreLst = []
    for h in range(h_len):
        for w in range(w_len):
            row = h * tile_size
            col = w * tile_size
            if PREDS[h * w_len + w] > 0 and img[row:row + tile_size, col:col + tile_size].mean() < 200:
                label_image(PREDS[h * w_len + w], img, row, col)
                tempLst = [row+tile_size//2, col+tile_size//2]
                centreLst.append(tempLst)
    centreLst2Array = np.array(centreLst)
    y_pred = DBSCAN(eps=1.5*tile_size, min_samples=1).fit_predict(centreLst2Array)

    for idx in range(len(centreLst2Array)):
        # print(y_pred[idx], end='')
        cv2.putText(img, str(y_pred[idx]), (centreLst2Array[idx][1], centreLst2Array[idx][0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    numCluster = np.max(y_pred) + 1
    for idx in range(numCluster):
        top = (np.min(centreLst2Array[y_pred == idx], axis=0))[0] - tile_size // 2
        down = (np.max(centreLst2Array[y_pred == idx], axis=0))[0] + tile_size // 2
        left = (np.min(centreLst2Array[y_pred == idx], axis=0))[1] - tile_size // 2
        right = (np.max(centreLst2Array[y_pred == idx], axis=0))[1] + tile_size // 2
        cv2.rectangle(img, (left, top), (right, down), (0, 255, 0), 2)
        img = block(img, top, down, left, right, centreLst2Array)

    return img


if __name__ == '__main__':
    for i in range(len(df)):
        file = df[i:i + 1]
        dataset = PANDADataset(file, tile_size, n_tiles)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        LOGITS = []
        LOGITS2 = []
        with torch.no_grad():
            for data in tqdm(loader):
                data = data.to(device)
                with torch.no_grad():
                    logits = models[0](data)
                LOGITS.append(logits)

        LOGITS = torch.cat(LOGITS).sigmoid().cpu()
        PREDS = LOGITS.sum(1).round().numpy()

        row = file.iloc[0]
        img_id = row.image_id
        grade = row.isup_grade

        tiff_file = os.path.join(image_dir, f'{img_id}.tiff')
        image = skimage.io.MultiImage(tiff_file)[1]
        gen = gen_image(image)
        skimage.io.imsave(f'{img_id}_{grade}_{tile_size}_color_block.jpg', gen)

        h, w = get_size(image)
        out = PREDS.reshape(h, w)
        print(out)
        print(out.shape)

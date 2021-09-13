#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets import simple_extractor_dataset
import importlib

simple_extractor_dataset = importlib.reload(simple_extractor_dataset)
from datasets.simple_extractor_dataset import SimpleFolderDataset

from sklearn.neighbors import KDTree

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


spacedb = KDTree(positions)

class_types = {
    'Upper-Clothes': [5, 6, 7, 10],
    'Lower-Clothes': [9, 10, 12],
    'Hat': [1],
    'Gloves': [3],
    'Sunglasses': [4]
}
positions = np.array([(0, 0, 0), (51, 51, 0),
                      (255, 235, 205), (255, 228, 196), (255, 222, 173), (245, 222, 179), (222, 184, 135),
                      (210, 180, 140), (188, 143, 143), (244, 164, 96), (218, 165, 32), (205, 133, 63), (210, 105, 30),
                      (139, 69, 19), (160, 82, 45), (165, 42, 42),

                      (139, 0, 0), (128, 0, 0), (255, 248, 220), (255, 255, 204), (255, 255, 153), (255, 255, 102),
                      (255, 255, 0),
                      # (204,204,0), (153,153,0),

                      (0, 191, 255), (30, 144, 255), (100, 149, 237), (123, 104, 238), (106, 90, 205), (72, 61, 139),
                      (65, 105, 225), (0, 0, 255), (0, 0, 205), (0, 0, 128), (25, 25, 112),

                      (169, 169, 169), (176, 196, 222), (135, 206, 235), (70, 130, 180),
                      # (128,128,128), (105,105,105),
                      # (119,136,153), (112,128,144), (47,79,79),

                      (50, 205, 50), (0, 255, 0), (34, 139, 34), (0, 128, 0),
                      # (0,100,0),
                      (173, 255, 47), (154, 205, 50), (0, 250, 154), (144, 238, 144), (152, 251, 152), (60, 179, 113),
                      (32, 178, 170), (46, 139, 87),
                      # (128,128,0), (85,107,47), (107,142,35),

                      (255, 160, 122), (250, 128, 114), (233, 150, 122), (240, 128, 128), (205, 92, 92), (220, 20, 60),
                      (178, 34, 34), (255, 0, 0), (255, 99, 71), (255, 69, 0), (219, 112, 147),
                      (211, 211, 211), (220, 220, 220), (230, 230, 250), (255, 255, 255), (255, 250, 250),
                      (240, 255, 240), (245, 255, 250), (240, 255, 255), (253, 245, 230), (255, 250, 240),
                      (255, 255, 240), (240, 248, 255)])

spacedb = KDTree(positions)

rgb_col_dict = {
    (0, 0, 0): 'Black',
    (51, 51, 0): 'Black',
    # (170, 110, 140) : 'Brown',

    (255, 235, 205): 'Brown',
    (255, 228, 196): 'Brown',
    (255, 222, 173): 'Brown',
    (245, 222, 179): 'Brown',
    (222, 184, 135): 'Brown',
    (210, 180, 140): 'Brown',
    (188, 143, 143): 'Brown',
    (244, 164, 96): 'Brown',
    (218, 165, 32): 'Brown',
    (205, 133, 63): 'Brown',
    (210, 105, 30): 'Brown',
    (139, 69, 19): 'Brown',
    (160, 82, 45): 'Brown',
    (165, 42, 42): 'Brown',

    (139, 0, 0): 'Yellow',
    (128, 0, 0): 'Yellow',
    (255, 248, 220): 'Yellow',
    (255, 255, 204): 'Yellow',
    (255, 255, 153): 'Yellow',
    (255, 255, 102): 'Yellow',
    (255, 255, 0): 'Yellow',
    # (204, 204, 0) : 'Yellow',
    # (153, 153, 0) : 'Yellow',

    (0, 191, 255): 'Blue',
    (30, 144, 255): 'Blue',
    (100, 149, 237): 'Blue',

    # (95, 158, 160) : 'Blue',
    (123, 104, 238): 'Blue',
    (106, 90, 205): 'Blue',
    (72, 61, 139): 'Blue',
    (65, 105, 225): 'Blue',
    (0, 0, 255): 'Blue',
    (0, 0, 205): 'Blue',
    (0, 0, 128): 'Blue',
    (25, 25, 112): 'Blue',

    (169, 169, 169): 'Gray',
    (176, 196, 222): 'Gray',
    (135, 206, 235): 'Gray',
    (70, 130, 180): 'Gray',
    # (128, 128, 128) : 'Gray',
    # (105, 105, 105) : 'Gray',
    # (119, 136, 153) : 'Gray',
    # (112, 128, 144) : 'Gray',
    # (47, 79, 79) : 'Gray',
    (50, 205, 50): 'Green',
    (0, 255, 0): 'Green',
    (34, 139, 34): 'Green',
    (0, 128, 0): 'Green',
    # (0, 100, 0) : 'Green',
    (173, 255, 47): 'Green',
    (154, 205, 50): 'Green',
    (0, 250, 154): 'Green',
    (144, 238, 144): 'Green',
    (152, 251, 152): 'Green',
    (60, 179, 113): 'Green',
    (32, 178, 170): 'Green',
    (46, 139, 87): 'Green',
    # (128, 128, 0) : 'Green',
    # (85, 107, 47) : 'Green',
    # (107, 142, 35) : 'Green',
    (255, 160, 122): 'Red',
    (250, 128, 114): 'Red',
    (233, 150, 122): 'Red',
    (240, 128, 128): 'Red',
    (205, 92, 92): 'Red',
    (220, 20, 60): 'Red',
    (178, 34, 34): 'Red',
    (255, 0, 0): 'Red',
    (255, 99, 71): 'Red',
    (255, 69, 0): 'Red',
    (219, 112, 147): 'Red',
    (211, 211, 211): 'White',
    (220, 220, 220): 'White',
    (230, 230, 250): 'White',
    (255, 255, 255): 'White',
    (255, 250, 250): 'White',
    (240, 255, 240): 'White',
    (245, 255, 250): 'White',
    (240, 255, 255): 'White',
    (253, 245, 230): 'White',
    (255, 250, 240): 'White',
    (255, 255, 240): 'White',
    (240, 248, 255): 'White'
}

def get_nearest_simple_color_rgb(rgb):
    querycolor = rgb
    dist, index = spacedb.query(querycolor)
    if type(rgb) is list:
        positions_res = [positions[i] for i in index]
    else:
        return positions[index]
    return positions_res

def dominant_color(colors):
    nearest_colors_list = get_nearest_simple_color_rgb(colors)
    freq = {}
    for item in nearest_colors_list:
        item = tuple(map(tuple,item))[0]
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    freq = sorted(freq, key=freq.get, reverse=True)
    if len(freq)<3:
        for i in range(len(freq)+1, 4):
            freq.append(freq[0])
    return freq[0], freq[1], freq[2]

def get_target_pixel_colors(result_as_np_array, class_type_name, img, coords):
    list_colors = []
    lis = np.array(class_types[class_type_name])
    res = np.hstack([(np.where(result_as_np_array == x)) for x in lis])
    rows, columns = res[0], res[1]
    types_clothes = []
    for r, c in zip(rows, columns):
        if result_as_np_array[r][c] not in types_clothes:
            types_clothes.append(result_as_np_array[r][c])
        bgr = img[r, c]
        list_colors.append([bgr[2], bgr[1], bgr[0]])
    if list_colors == []:
        return None
    return dominant_color(list_colors), types_clothes

def get_target_object(result_as_np_array, class_type_name, img, coords):
    target_pixel_info = get_target_pixel_colors(result_as_np_array, class_type_name, img, coords)
    if target_pixel_info is None:
        return None
    dominant_colors = target_pixel_info[0]
    types_clothes = target_pixel_info[1]
    global rgb_col_dict
    color1 = rgb_col_dict[dominant_colors[0]]
    color2 = rgb_col_dict[dominant_colors[1]]
    color3 = rgb_col_dict[dominant_colors[2]]
    coords1 = {
        'x1': int(coords[0]),
        'y1': int(coords[1]),
        'x2': int(coords[2]),
        'y2': int(coords[3])
    }
    coords2 = {
        'top-left': [int(coords[0]), int(coords[1])],
        'bottom-right': [int(coords[2]), int(coords[3])]
    }
    return {
        'class': class_type_name,
        'cloth_type_list': [dataset_settings['lip']['label'][x] for x in types_clothes],
        'confidence': 100,
        'coordinates': coords1,
        'coords': coords2,
        'color1': color1,
        'color2': color2,
        'color3': color3
    }

def get_instance_objects(result_as_np_array, img, coords):
    global class_types
    class_types_list = class_types.keys()
    objects = []
    for class_type_name in class_types_list:
        res = get_target_object(result_as_np_array, class_type_name, img, coords)
        if res != None:
            objects.append(res)
    return objects


def main(**args):
    gpus = [int(i) for i in args['gpu'].split(',')]
    assert len(gpus) == 1
    if not args['gpu'] == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']

    num_classes = dataset_settings[args['dataset']]['num_classes']
    input_size = dataset_settings[args['dataset']]['input_size']

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args['model_restore'])['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(list_im=args['img_list'], input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    objects = {}
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            result_as_np_array = np.asarray(parsing_result, dtype=np.uint8)
            key = img_name[:-2]
            if key in objects.keys():
                objects[key] += get_instance_objects(result_as_np_array, args['img_list'][img_name], args['coords'][img_name])
            else:
                objects[key] = get_instance_objects(result_as_np_array, args['img_list'][img_name], args['coords'][img_name])
    return objects

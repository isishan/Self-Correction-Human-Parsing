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

from scipy.spatial import KDTree

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


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='checkpoints/final.pth',
                        help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='new_images', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='out', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def print_pic(coords, img_path, colors):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

    im = Image.open(img_path)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    rect = patches.Rectangle(coords[0], coords[1][0] - coords[0][0], coords[1][1] - coords[0][1], linewidth=1,
                             edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    colors = [get_nearest_simple_color_rgb(x)[1] for x in colors]
    ax.annotate(str(colors), coords[0], color='black', weight='bold', fontsize=10, ha='center', va='center')
    plt.savefig(img_path[:-3] + 'new.png', dpi=300, bbox_inches="tight")
    plt.show()


names = ['Black', 'Black',
         'Brown', 'Brown', 'Brown',
         'Yellow', 'Yellow', 'Yellow', 'Yellow', 'Yellow', 'Yellow',
         'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue',
         'Blue', 'Blue', 'Blue',
         'Grey', 'Grey', 'Grey', 'Grey', 'Grey', 'Grey', 'Grey',
         'Green', 'Green', 'Green', 'Green', 'Green', 'Green', 'Green', 'Green', 'Green', 'Green', 'Green', 'Green',
         'Green', 'Green', 'Green', 'Green',
         'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red',
         'White', 'White', 'White', 'White', 'White', 'White', 'White', 'White', 'White', 'White', 'White', 'White']
positions = [(0, 0, 0), (51, 51, 0),
             (170, 110, 140), (139, 0, 0), (128, 0, 0),
             (255, 255, 204), (255, 255, 153), (255, 255, 102), (255, 255, 0), (204, 204, 0), (153, 153, 0),
             (176, 224, 230), (135, 206, 235), (0, 191, 255), (176, 196, 222), (30, 144, 255), (100, 149, 237),
             (70, 130, 180), (95, 158, 160), (123, 104, 238), (106, 90, 205), (72, 61, 139), (65, 105, 225),
             (0, 0, 255), (0, 0, 205), (0, 0, 128), (25, 25, 112),
             (192, 192, 192), (169, 169, 169), (128, 128, 128), (105, 105, 105), (119, 136, 153), (112, 128, 144),
             (47, 79, 79),
             (50, 205, 50), (0, 255, 0), (34, 139, 34), (0, 128, 0), (0, 100, 0), (173, 255, 47), (154, 205, 50),
             (0, 250, 154), (144, 238, 144), (152, 251, 152), (60, 179, 113), (32, 178, 170), (46, 139, 87),
             (128, 128, 0), (85, 107, 47), (107, 142, 35),
             (255, 160, 122), (250, 128, 114), (233, 150, 122), (240, 128, 128), (205, 92, 92), (220, 20, 60),
             (178, 34, 34), (255, 0, 0), (255, 99, 71), (255, 69, 0), (219, 112, 147),
             (211, 211, 211), (220, 220, 220), (230, 230, 250), (255, 255, 255), (255, 250, 250), (240, 255, 240),
             (245, 255, 250), (240, 255, 255), (253, 245, 230), (255, 250, 240), (255, 255, 240), (240, 248, 255)
             ]

spacedb = KDTree(positions)


def get_nearest_simple_color_rgb(rgb):
    global avg_time1, avg_time2, avg_time3
    start3 = time.time()

    start1 = time.time()
    querycolor = rgb

    dist, index = spacedb.query(querycolor)
    end1 = time.time()
    start2 = time.time()

    # print('The color %r is closest to %s.'%(querycolor, names[index]))
    if type(rgb) is list:
        positions_res = [positions[i] for i in index]
        names_res = [names[i] for i in index]
    else:
        return positions[index], names[index]
    end = time.time()
    avg_time1 = (avg_time1 + (end1 - start1)) / 2
    avg_time2 = (avg_time2 + (end - start2)) / 2
    avg_time3 = (avg_time3 + (end - start3)) / 2
    return positions_res, names_res
    # return positions[index], names[index]
    # print('The color %r is closest to %s.'%(querycolor, names[index]))


def dominant_color(colors):
    nearest_colors_list = []
    # for i in colors:
    #     nearest_colors_list.append(get_nearest_simple_color_rgb(i)[0])
    nearest_colors_list = get_nearest_simple_color_rgb(colors)[0]
    freq = {}
    for item in nearest_colors_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    freq = sorted(freq, key=freq.get, reverse=True)
    # print(freq)
    return freq[0]


def dominant_color_2(nearest_colors_list):
    freq = {}
    for item in nearest_colors_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    freq = sorted(freq, key=freq.get, reverse=True)
    # print(freq)
    return freq[0]


class_dict = {
    'Upper': [5, 6, 7, 10],
    'Lower': [9, 10, 12]
}
import time

avg_time1, avg_time2, avg_time3 = 0, 0, 0


def get_target_pixels(result_as_np_array, class_name, img):
    # img_path = '/content/Self-Correction-Human-Parsing/new_images/' + img_name
    # im = Image.open(img_path)
    # pix = im.load()
    list_colors = []
    # cv2_imshow(img)
    # for x_, x in enumerate(result_as_np_array):
    #   for y_, y in enumerate(x):
    #     if result_as_np_array[x_, y_] in class_dict[class_name]:
    #       bgr = img[x_,y_]
    #       list_colors.append([bgr[2], bgr[1], bgr[0]])

    lis = np.array(class_dict[class_name])
    res = [(np.where(result_as_np_array == x)) for x in lis]
    res = (np.hstack(res))
    rows, columns = res[0], res[1]
    for r, c in zip(rows, columns):
        bgr = img[r, c]
        list_colors.append([bgr[2], bgr[1], bgr[0]])
    return list_colors
    # if list_colors == []:
    #     return None
    # color1 = get_nearest_simple_color_rgb(dominant_color(list_colors))
    # coords1 = {
    #     'x1': int(coords[0]),
    #     'y1': int(coords[1]),
    #     'x2': int(coords[2]),
    #     'y2': int(coords[3])
    # }
    # coords2 = {
    #     'top-left': [int(coords[0]), int(coords[1])],
    #     'bottom-right': [int(coords[2]), int(coords[3])]
    # }
    # return {
    #     'class': class_name,
    #     'confidence': 100,
    #     'coordinates': coords1,
    #     'coords': coords2,
    #     'color1': color1[1]
    # }


def get_objects(result_as_np_array, img):
    # class_names = ['Upper', 'Lower']
    # objects = []
    # for class_name in class_names:
    #     res = get_target_pixels(result_as_np_array, class_name, img, coords)
    #     if res != None:
    #         objects.append(res)
    # return objects

    class_names = ['Upper', 'Lower']
    objects = []
    sizes = [0]
    consolidated_target_colors = []
    for class_name in class_names:
        instance_target_colors = get_target_pixels(result_as_np_array, class_name, img)
        sizes.append(sizes[-1] + len(instance_target_colors))
        consolidated_target_colors += instance_target_colors
    return consolidated_target_colors, sizes


def get_final_objects(input_obj):
    main_list_pixels = []
    for key, val in input_obj.items():
        # print(key, len(val))
        for v in val:
            # print(len(v))
            main_list_pixels += v[0]
    nearest_colors_list = get_nearest_simple_color_rgb(main_list_pixels)[0]
    val1 = {}
    for key, val in input_obj.items():
        for v in val:
            if key in val1:
                val1[key].append(dominant_color_2(nearest_colors_list[v[1][0]:v[1][1]]))
            else:
                val1[key] = [dominant_color_2(nearest_colors_list[v[1][0]:v[1][1]])]
    # print(val1)
    return input_obj


def main(**args):
    # os.chdir('content/Self-Correction-Human-Parsing/')
    # print(os.getcwd())
    # args = get_arguments()
    gpus = [int(i) for i in args['gpu'].split(',')]
    assert len(gpus) == 1
    if not args['gpu'] == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']

    num_classes = dataset_settings[args['dataset']]['num_classes']
    input_size = dataset_settings[args['dataset']]['input_size']
    label = dataset_settings[args['dataset']]['label']
    # print("Evaluating total class number {} with {}".format(num_classes, label))

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
    # dataset = SimpleFolderDataset(root=args['input_dir'], input_size=input_size, transform=transform)
    dataset = SimpleFolderDataset(list_im=args['img_list'], input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    palette = get_palette(num_classes)
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
            parsing_result_path = os.path.join(args['output_dir'], img_name + '.png')
            result_as_np_array = np.asarray(parsing_result, dtype=np.uint8)
            key = img_name[:-2]
            # print()
            # print(args['coords'])
            # print(args['img_list'])
            # print(key)
            instance_res, instance_sizes = get_objects(result_as_np_array, args['img_list'][img_name])
            if key in objects.keys():
                objects[key].append(instance_res, instance_sizes, args['coords'][img_name])
            else:
                objects[key] = [[instance_res, instance_sizes, args['coords'][img_name]]]

            # output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            # output_img.putpalette(palette)
            # output_img.save(parsing_result_path)
            # if args.logits:
            #     logits_result_path = os.path.join(output_dir, img_name[:-4] + '.npy')
            #     np.save(logits_result_path, logits_result)
    objects = get_final_objects(objects)
    # print("AVergae time 1", avg_time1)
    # print("AVergae time 2", avg_time2)
    # print("AVergae time 3", avg_time3)
    return objects

# if __name__ == '__main__':
#     main()




# -*- coding: utf-8 -*-
'''
@File    :   visualizer.py
@Time    :   2022/04/05 11:39:33
@Author  :   Shilong Liu 
@Contact :   liusl20@mail.tsinghua.edu.cn; slongliu86@gmail.com
Modified from COCO evaluator
'''

import os, sys
from textwrap import wrap
import torch
import numpy as np
import cv2
import datetime

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
from matplotlib import transforms

def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)

class ColorMap():
    def __init__(self, basergb=[255,255,0]):
        self.basergb = np.array(basergb)
    def __call__(self, attnmap):
        # attnmap: h, w. np.uint8.
        # return: h, w, 4. np.uint8.
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1) # h, w, 3
        attn1 = attnmap.copy()[..., None] # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res


class COCOVisualizer():
    def __init__(self) -> None:
        pass

    def visualize(self, img, tgt, iou=None, caption=None, dpi=120, savedir=None, show_in_console=True, num=0,
                  size_image=(10, 10), view_all=False):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        # """
        # my_dpi = 132
        # plt.figure(figsize=(1920/my_dpi, 1080/my_dpi), dpi=my_dpi)

        if dpi is None:
            plt.figure(figsize=size_image)
        else:  
            plt.figure(dpi=dpi)
            
        ax = plt.gca()
        img = renorm(img).permute(1, 2, 0)
        ax.imshow(img)

        self.addtgt(tgt, iou=iou, view_all=view_all)
        if show_in_console:
            plt.show()
        
        plt.axis('off')

        if savedir is not None:
            if caption is None and iou is None:
                    savename = '{}_imageID:{}__{}.png'.format(num, int(tgt['image_id']), 
                                    str(datetime.datetime.now()).replace(' ', '-'))
            elif caption is None and iou is not None:
                    savename = '{}_imageID:{}__IoU:{}__{}.png'.format(num, int(tgt['image_id']), 
                                    "{:.3f}".format(iou.item()), str(datetime.datetime.now()).replace(' ', '-'))
            elif caption is not None and iou is None:
                    savename = '{}_imageID:{}_({})__{}.png'.format(num, int(tgt['image_id']), 
                                    caption, str(datetime.datetime.now()).replace(' ', '-'))
            else:
                    savename = '{}_imageID:{}_({})__IoU:{}__{}.png'.format(num, int(tgt['image_id']), 
                                    caption, "{:.3f}".format(iou.item()), str(datetime.datetime.now()).replace(' ', '-'))
            print("savename: {}".format(savename))
            # os.makedirs(os.path.dirname(savename), exist_ok=True)
            plt.savefig(os.path.join(savedir, savename), pad_inches=0, bbox_inches='tight')
        plt.close()

    def addtgt(self, tgt, iou=None, view_all=False):
        """
        - tgt: dict. args:
            - boxes: num_boxes, 4. xywh, [0,1].
            - box_label: num_boxes.
        """
        assert 'boxes' in tgt
        ax = plt.gca()
        H, W = tgt['size'].tolist() 
        # numbox = tgt['boxes'].shape[0]
        fsize = 9 # or 6 if we do downsampling

        color = []
        polygons = []
        boxes = []

        if view_all is False:
            for box in tgt['boxes'].cpu():
                unnormbbox = box * torch.Tensor([W, H, W, H])
                # unnormbbox = box
                unnormbbox[:2] -= unnormbbox[2:] / 2
                [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
                boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
                poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                np_poly = np.array(poly).reshape((4,2))
                polygons.append(Polygon(np_poly))
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=0.5)
            ax.add_collection(p)


            if 'box_label' in tgt:
                # assert len(tgt['box_label']) == numbox, f"{len(tgt['box_label'])} = {numbox}, "
                for idx, bl in enumerate(tgt['box_label']):
                    _string = str(bl)
                    if iou is not None:
                        if idx == 0:
                            _string += " (GT)"
                        elif idx == 1:
                            _string += " (Pred)"

                    bbox_x, bbox_y, bbox_w, bbox_h = boxes[idx]
                    # ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 1})
                    ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[idx], 'alpha': 0.6, 'pad': 1}, fontsize=fsize)
        else:
            for idx, bbox_key in enumerate(tgt['boxes'].keys()):  
                bbox_coord = tgt['boxes'][bbox_key]
                for bbox in bbox_coord:
                    bbox = bbox.cpu()
                    unnormbbox = bbox * torch.Tensor([W, H, W, H])
                    # unnormbbox = bbox
                    unnormbbox[:2] -= unnormbbox[2:] / 2
                    [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
                    boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4,2))
                    polygons.append(Polygon(np_poly))
                    if idx == 0:
                        color.append((0,1,0))
                    else:
                        color.append((1,0,0))

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.1)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=0.5)
            ax.add_collection(p) 

            limit_assign = True
            if tgt['boxes']['gt'].size()[0] > tgt['boxes']['pred'].size()[0]:
                limit_assign = tgt['boxes']['pred'].size()[0]
            elif tgt['boxes']['gt'].size()[0] < tgt['boxes']['pred'].size()[0]:
                limit_assign = tgt['boxes']['gt'].size()[0] 
            else:
                limit_assign = None

            if 'box_label' in tgt:
                count_idx = 0
                count_assign = 0
                # assert len(tgt['box_label']) == numbox, f"{len(tgt['box_label'])} = {numbox}, "
                for idx, bl_key in enumerate(tgt['box_label']):
                    bl = tgt['box_label'][bl_key]
                    for label in bl:
                        if limit_assign is None or (count_assign < limit_assign):
                            iou_value = tgt['iou'][count_assign].item()
                            if idx == 0:
                                _string = f"{count_assign+1}.{str(label)}: {iou_value:.2f}"
                            elif idx == 1:
                                _string = f"{count_assign+1}.{str(label)}: {iou_value:.2f}"
                            count_assign += 1

                        else:
                            _string = f"0.{str(label)}"

                        bbox_x, bbox_y, bbox_w, bbox_h = boxes[count_idx]
                        # ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': 'yellow', 'alpha': 1.0, 'pad': 1})
                        ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[count_idx], 'alpha': 0.6, 'pad': 1}, fontsize=fsize)
                        count_idx += 1
                    count_assign = 0


# class COCOVisualizer():
#     def __init__(self) -> None:
#         pass

#     def visualize(self, img, tgt, iou=None, caption=None, dpi=120, savedir=None, show_in_console=True, num=0,
#                   size_image=(10, 10), view_all=False):
#         """
#         img: tensor(3, H, W)
#         tgt: make sure they are all on cpu.
#             must have items: 'image_id', 'boxes', 'size'
#         """
#         img = renorm(img)
#         img_pil = self.tensor_to_pil(img)

#         self.addtgt(tgt, iou=iou, view_all=view_all)

#         if show_in_console:
#             img_pil.show()

#         if savedir is not None:
#             if caption is None and iou is None:
#                     savename = '{}_imageID:{}__{}.png'.format(num, int(tgt['image_id']), 
#                                     str(datetime.datetime.now()).replace(' ', '-'))
#             elif caption is None and iou is not None:
#                     savename = '{}_imageID:{}__IoU:{}__{}.png'.format(num, int(tgt['image_id']), 
#                                     "{:.3f}".format(iou.item()), str(datetime.datetime.now()).replace(' ', '-'))
#             elif caption is not None and iou is None:
#                     savename = '{}_imageID:{}_({})__{}.png'.format(num, int(tgt['image_id']), 
#                                     caption, str(datetime.datetime.now()).replace(' ', '-'))
#             else:
#                     savename = '{}_imageID:{}_({})__IoU:{}__{}.png'.format(num, int(tgt['image_id']), 
#                                     caption, "{:.3f}".format(iou.item()), str(datetime.datetime.now()).replace(' ', '-'))
#             print("savename: {}".format(savename))
#             # os.makedirs(os.path.dirname(savename), exist_ok=True)
#             img_pil.save(os.path.join(savedir, savename))
    
#     def tensor_to_pil(self, img):
#         img = img.mul(255).byte().numpy().transpose(1, 2, 0)
#         return Image.fromarray(img)
    
#     def addtgt(self, tgt, iou=None, view_all=False):
#         """
#         - tgt: dict. args:
#             - boxes: num_boxes, 4. xywh, [0,1].
#             - box_label: num_boxes.
#         """
#         assert 'boxes' in tgt

#         H, W = tgt['size'].tolist()
#         img_pil = Image.new("RGB", (W, H))
#         draw = ImageDraw.Draw(img_pil)
#         font = ImageFont.load_default()

#         color = []

#         if view_all is False:
#             for idx, box in enumerate(tgt['boxes'].cpu()):
#                 unnormbbox = box * torch.Tensor([W, H, W, H])
#                 unnormbbox[:2] -= unnormbbox[2:] / 2
#                 [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
#                 c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
#                 color.append(c)

#                 draw.rectangle([bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h], outline=tuple(map(int, c)))

#                 if 'box_label' in tgt:
#                     _string = str(tgt['box_label'][idx])
#                     if iou is not None:
#                         if idx == 0:
#                             _string += " (GT)"
#                         elif idx == 1:
#                             _string += " (Pred)"

#                     draw.text((bbox_x, bbox_y), _string, fill='black', font=font)

#         else:
#             for idx, bbox_key in enumerate(tgt['boxes'].keys()):
#                 bbox_coord = tgt['boxes'][bbox_key]
#                 for bbox in bbox_coord:
#                     bbox = bbox.cpu()
#                     unnormbbox = bbox * torch.Tensor([W, H, W, H])
#                     unnormbbox[:2] -= unnormbbox[2:] / 2
#                     [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
#                     if idx == 0:
#                         c = (0, 255, 0)  # Green for GT
#                     else:
#                         c = (255, 0, 0)  # Red for Pred

#                     color.append(c)
#                     print(bbox_x, bbox_y, bbox_w, bbox_h)
#                     draw.rectangle([bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h], outline=tuple(map(int, c)))

#                 limit_assign = True
#                 if tgt['boxes']['gt'].size()[0] > tgt['boxes']['pred'].size()[0]:
#                     limit_assign = tgt['boxes']['pred'].size()[0]
#                 elif tgt['boxes']['gt'].size()[0] < tgt['boxes']['pred'].size()[0]:
#                     limit_assign = tgt['boxes']['gt'].size()[0]
#                 else:
#                     limit_assign = None
#                 print(tgt['boxes']['gt'], tgt['boxes']['pred'])

#                 if 'box_label' in tgt:
#                     count_idx = 0
#                     count_assign = 0
#                     for idx, bl_key in enumerate(tgt['box_label']):
#                         bl = tgt['box_label'][bl_key]
#                         for label in bl:
#                             if limit_assign is None or (count_assign < limit_assign):
#                                 iou_value = tgt['iou'][count_assign].item()
#                                 if idx == 0:
#                                     _string = f"{count_assign+1}.{str(label)}: {iou_value:.2f}"
#                                 elif idx == 1:
#                                     _string = f"{count_assign+1}.{str(label)}: {iou_value:.2f}"
#                                 count_assign += 1
#                             else:
#                                 _string = f"0.{str(label)}"

#                             draw.text((bbox_x, bbox_y), _string, fill='black', font=font)
#                             count_idx += 1
#                         count_assign = 0
#         self.img_pil = img_pil



import torch
import warnings
torch.autograd.set_detect_anomaly(True)
warnings.simplefilter("ignore")
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import json
import argparse
import timm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from utils.config_utils import load_yaml
from vis_utils import ImgLoader

def build_model(pretrainewd_path: str,
                img_size: int, 
                fpn_size: int, 
                num_classes: int,
                num_selects: dict,
                use_fpn: bool = True, 
                use_selection: bool = True,
                use_combiner: bool = True, 
                comb_proj_size: int = None):
    from models.pim_module.pim_module_eval.py import PluginMoodel

    model = \
        PluginMoodel(img_size = img_size,
                     use_fpn = use_fpn,
                     fpn_size = fpn_size,
                     proj_type = "Linear",
                     upsample_type = "Conv",
                     use_selection = use_selection,
                     num_classes = num_classes,
                     num_selects = num_selects, 
                     use_combiner = use_combiner,
                     comb_proj_size = comb_proj_size)

    if pretrainewd_path != "":
        ckpt = torch.load(pretrainewd_path)
        model.load_state_dict(ckpt['model'])
    
    model.eval()

    return model
@torch.no_grad()
def sum_all_out(out, sum_type="softmax"):
    target_layer_names = \
    ['layer1', 'layer2', 'layer3', 'layer4',
    'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4', 
    'comb_outs']

    sum_out = None
    for name in target_layer_names:
        if name != "comb_outs":
            tmp_out = out[name].mean(1)
        else:
            tmp_out = out[name]
        
        if sum_type == "softmax":
            tmp_out = torch.softmax(tmp_out, dim=-1)
        if sum_out is None:
            sum_out = tmp_out
        else:
            sum_out = sum_out + tmp_out # note that use '+=' would cause inplace error
    return sum_out

if __name__ == "__main__":
    # ===== 0. get setting =====
    parser = argparse.ArgumentParser("Visualize SwinT Large")
    parser.add_argument("-pr", "--pretrained_root", type=str, 
        help="contain {pretrained_root}/best.pt, {pretrained_root}/config.yaml")
    parser.add_argument("-ir", "--image_root", type=str)
    args = parser.parse_args()

    load_yaml(args, args.pretrained_root + "/config.yaml")

    # ===== 1. build model =====
    model = build_model(pretrainewd_path = args.pretrained_root + "/best.pt",
                        img_size = args.data_size, 
                        fpn_size = args.fpn_size, 
                        num_classes = args.num_classes,
                        num_selects = args.num_selects)
    model.cuda()

    img_loader = ImgLoader(img_size = args.data_size)

    cls_folders = os.listdir(args.image_root)
    cls_folders.sort()
    top1, top3, top5 = 0, 0, 0
    total = 0
    n_samples = 0

    flycatcher = np.zeros([8, 8], dtype=np.float32) # 36~42
    gull = np.zeros([9, 9], dtype=np.float32) # 58~65
    kingfisher = np.zeros([6, 6], dtype=np.float32) # 78~82
    sparrow = np.zeros([22, 22], dtype=np.float32) #112~132
    tern = np.zeros([8, 8], dtype=np.float32) # 140~146
    vireo = np.zeros([8, 8], dtype=np.float32) # 150~156
    warbler = np.zeros([26, 26], dtype=np.float32) # 157~181
    woodpecker = np.zeros([7, 7], dtype=np.float32) # 186~191
    wren = np.zeros([26, 26], dtype=np.float32) # 192~198

    for ci, cf in enumerate(cls_folders):
        n_samples += len(os.listdir(args.image_root + "/" + cf))
    pbar = tqdm.tqdm(total=n_samples, ascii=True)
    wrongs = {}
    for ci, cf in enumerate(cls_folders):
        files = os.listdir(args.image_root + "/" + cf)
        imgs = []
        img_paths = []
        update_n = 0
        for fi, f in enumerate(files):
            img_path = args.image_root + "/" + cf + "/" + f
            img_paths.append(img_path)
            img, ori_img = img_loader.load(img_path)
            img = img.unsqueeze(0) # add batch size dimension
            imgs.append(img)
            update_n += 1
            if (fi+1) % 32 == 0 or fi == len(files) - 1:    
                imgs = torch.cat(imgs, dim=0)
            else:
                continue
            with torch.no_grad():
                imgs = imgs.cuda()
                outs = model(imgs)
                sum_outs = sum_all_out(outs, sum_type="softmax") # softmax
                preds = torch.sort(sum_outs, dim=-1, descending=True)[1]
                for bi in range(preds.size(0)):
                    if preds[bi, 0] == ci:
                        top1 += 1
                        top3 += 1
                        top5 += 1
                    else:
                        if ci not in wrongs:
                            wrongs[ci] = []
                        wrongs[ci].append(img_paths[bi])

                    if preds[bi, 1] == ci or preds[bi, 2] == ci:
                        top3 += 1
                        top5 += 1

                    if preds[bi, 3] == ci or preds[bi, 4] == ci:
                        top5 += 1
                total += update_n

                basic_n = None
                if 36 <= ci <= 42:
                    basic_n = 36
                    for bi in range(preds.size(0)):
                        in_pred = int(preds[bi,0])-basic_n
                        if in_pred < 0 or in_pred >= flycatcher.shape[0]-1:
                            in_pred = flycatcher.shape[0]-1
                        flycatcher[ci-basic_n][in_pred] += 1
                elif 58 <= ci <= 65:
                    basic_n = 58
                    for bi in range(preds.size(0)):
                        in_pred = int(preds[bi,0])-basic_n
                        if in_pred < 0 or in_pred >= gull.shape[0]-1:
                            in_pred = gull.shape[0]-1
                        gull[ci-basic_n][in_pred] += 1
                elif 78 <= ci <= 82:
                    basic_n = 78
                    for bi in range(preds.size(0)):
                        in_pred = int(preds[bi,0])-basic_n
                        if in_pred < 0 or in_pred >= kingfisher.shape[0]-1:
                            in_pred = kingfisher.shape[0]-1
                        kingfisher[ci-basic_n][in_pred] += 1
                elif 112 <= ci <= 132:
                    basic_n = 112
                    for bi in range(preds.size(0)):
                        in_pred = int(preds[bi,0])-basic_n
                        if in_pred < 0 or in_pred >= sparrow.shape[0]-1:
                            in_pred = sparrow.shape[0]-1
                        sparrow[ci-basic_n][in_pred] += 1
                elif 140 <= ci <= 146:
                    basic_n = 140
                    for bi in range(preds.size(0)):
                        in_pred = int(preds[bi,0])-basic_n
                        if in_pred < 0 or in_pred >= tern.shape[0]-1:
                            in_pred = tern.shape[0]-1
                        tern[ci-basic_n][in_pred] += 1
                elif 150 <= ci <= 156:
                    basic_n = 150
                    for bi in range(preds.size(0)):
                        in_pred = int(preds[bi,0])-basic_n
                        if in_pred < 0 or in_pred >= vireo.shape[0]-1:
                            in_pred = vireo.shape[0]-1
                        vireo[ci-basic_n][in_pred] += 1
                elif 157 <= ci <= 181:
                    basic_n = 157
                    for bi in range(preds.size(0)):
                        in_pred = int(preds[bi,0])-basic_n
                        if in_pred < 0 or in_pred >= warbler.shape[0]-1:
                            in_pred = warbler.shape[0]-1
                        warbler[ci-basic_n][in_pred] += 1
                elif 186 <= ci <= 191:
                    basic_n = 186
                    for bi in range(preds.size(0)):
                        in_pred = int(preds[bi,0])-basic_n
                        if in_pred < 0 or in_pred >= woodpecker.shape[0]-1:
                            in_pred = woodpecker.shape[0]-1
                        woodpecker[ci-basic_n][in_pred] += 1
                elif 192 <= ci <= 198:
                    basic_n = 192
                    for bi in range(preds.size(0)):
                        in_pred = int(preds[bi,0])-basic_n
                        if in_pred < 0 or in_pred >= wren.shape[0]-1:
                            in_pred = wren.shape[0]-1
                        wren[ci-basic_n][in_pred] += 1
                    

            imgs = []
            img_paths = []
            top1_acc = round(top1 / total * 100, 3)
            top3_acc = round(top3 / total * 100, 3)
            top5_acc = round(top5 / total * 100, 3)
            if flycatcher.sum() != 0:
                flycatcher_acc = round(np.trace(flycatcher) / flycatcher.sum() * 100, 3)
                flycatcher_out = flycatcher[:, -1].sum()
            else:
                flycatcher_acc = -1
                flycatcher_out = -1
            
            if gull.sum() != 0:
                gull_acc = round(np.trace(gull) / gull.sum() * 100, 3)
                gull_out = gull[:, -1].sum()
            else:
                gull_acc = -1
                gull_out = -1

            if kingfisher.sum() != 0:
                kingfisher_acc = round(np.trace(kingfisher) / kingfisher.sum() * 100, 3)
                kingfisher_out = kingfisher[:, -1].sum()
            else:
                kingfisher_acc = -1
                kingfisher_out = -1
            
            if sparrow.sum() != 0:
                sparrow_acc = round(np.trace(sparrow) / sparrow.sum() * 100, 3)
                sparrow_out = sparrow[:, -1].sum()
            else:
                sparrow_acc = -1
                sparrow_out = -1

            if tern.sum() != 0:
                tern_acc = round(np.trace(tern) / tern.sum() * 100, 3)
                tern_out = tern[:, -1].sum()
            else:
                tern_acc = -1
                tern_out = -1

            if vireo.sum() != 0:
                vireo_acc = round(np.trace(vireo) / vireo.sum() * 100, 3)
                vireo_out = vireo[:, -1].sum()
            else:
                vireo_acc = -1
                vireo_out = -1
            
            if warbler.sum() != 0:
                warbler_acc = round(np.trace(warbler) / warbler.sum() * 100, 3)
                warbler_out = warbler[:, -1].sum()
            else:
                warbler_acc = -1
                warbler_out = -1

            if woodpecker.sum() != 0:
                woodpecker_acc = round(np.trace(woodpecker) / woodpecker.sum() * 100, 3)
                woodpecker_out = woodpecker[:, -1].sum()
            else:
                woodpecker_acc = -1
                woodpecker_out = -1


            if wren.sum() != 0:
                wren_acc = round(np.trace(wren) / wren.sum() * 100, 3)
                wren_out = wren[:, -1].sum()
            else:
                wren_acc = -1
                wren_out = -1

            msg = "top1: {}%, top3: {}%, top5: {}%".format(top1_acc, top3_acc, top5_acc)
            pbar.set_description(msg)
            pbar.update(update_n)
            update_n = 0
    pbar.close()

    msg = "\n=== evaluation result on CUB200-2011 ===\n\
top1: {}%, top3: {}%, top5: {}%, \n\
flycatcher:{}%  out:{}, \n\
gull:{}%,  out:{}, \n\
kingfisher:{}%  out:{}, \n\
sparrow:{}%  out:{}, \n\
tern:{}%  out:{}, \n\
vireo:{}%  out:{}, \n\
warbler:{}%  out:{}, \n\
woodpecker:{}%  out:{}, \n\
wren:{}%  out:{}".format(
                    top1_acc, top3_acc, top5_acc, 
                    flycatcher_acc, flycatcher_out,
                    gull_acc, gull_out,
                    kingfisher_acc, kingfisher_out,
                    sparrow_acc, sparrow_out,
                    tern_acc, tern_out,
                    vireo_acc, vireo_out,
                    warbler_acc, warbler_out,
                    woodpecker_acc, woodpecker_out,
                    wren_acc, wren_out
                )
    print(msg)

    # np.save('flycatcher.npy', flycatcher)
    # np.save('gull.npy', gull)
    # np.save('kingfisher.npy', kingfisher)
    # np.save('sparrow.npy', sparrow)
    # np.save('tern.npy', tern)
    # np.save('vireo.npy', vireo)
    # np.save('warbler.npy', warbler)
    # np.save('woodpecker.npy', woodpecker)
    # np.save('wren.npy', wren)

    # with open("wrongs_list.json", "w") as fjson:
    #     fjson.write(json.dumps(wrongs, indent=2))

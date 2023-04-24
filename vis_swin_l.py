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
import argparse
import timm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from utils.config_utils import load_yaml
from vis_utils import ImgLoader, get_cdict

global module_id_mapper
global features
global grads

def forward_hook(module: nn.Module, inp_hs, out_hs):
    global features, module_id_mapper
    layer_id = len(features) + 1
    module_id_mapper[module] = layer_id
    features[layer_id] = {}
    features[layer_id]["in"] = inp_hs
    features[layer_id]["out"] = out_hs
    # print('forward_hook, layer_id:{}, hs_size:{}'.format(layer_id, out_hs.size()))

def backward_hook(module: nn.Module, inp_grad, out_grad):
    global grads, module_id_mapper
    layer_id = module_id_mapper[module]
    grads[layer_id] = {}
    grads[layer_id]["in"] = inp_grad
    grads[layer_id]["out"] = out_grad
    # print('backward_hook, layer_id:{}, hs_size:{}'.format(layer_id, out_grad[0].size()))


def build_model(pretrainewd_path: str,
                img_size: int, 
                fpn_size: int, 
                num_classes: int,
                num_selects: dict,
                use_fpn: bool = True, 
                use_selection: bool = True,
                use_combiner: bool = True, 
                comb_proj_size: int = None):
    from models.pim_module import PluginMoodel

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

    ### hook original layer1~4
    model.backbone.layers[0].register_forward_hook(forward_hook)
    model.backbone.layers[0].register_full_backward_hook(backward_hook)
    model.backbone.layers[1].register_forward_hook(forward_hook)
    model.backbone.layers[1].register_full_backward_hook(backward_hook)
    model.backbone.layers[2].register_forward_hook(forward_hook)
    model.backbone.layers[2].register_full_backward_hook(backward_hook)
    model.backbone.layers[3].register_forward_hook(forward_hook)
    model.backbone.layers[3].register_full_backward_hook(backward_hook)
    ### hook original FPN layer1~4
    model.fpn_down.Proj_layer1.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer1.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer2.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer2.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer3.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer3.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer4.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer4.register_full_backward_hook(backward_hook)
    ### hook original FPN_UP layer1~4
    model.fpn_up.Proj_layer1.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer1.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer2.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer2.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer3.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer3.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer4.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer4.register_full_backward_hook(backward_hook)

    return model

def cal_backward(args, out, sum_type: str = "softmax"):
    assert sum_type in ["none", "softmax"]

    target_layer_names = ['layer1', 'layer2', 'layer3', 'layer4',
    'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4', 'comb_outs']

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

    with torch.no_grad():
        if args.use_label:
            print("use label as target class")
            pred_score = torch.softmax(sum_out, dim=-1)[0][args.label]
            backward_cls = args.label
        else:
            pred_score, pred_cls = torch.max(torch.softmax(sum_out, dim=-1), dim=-1)
            pred_score = pred_score[0]
            pred_cls = pred_cls[0]
            backward_cls = pred_cls

    print(sum_out.size())
    print("pred: {}, gt: {}, score:{}".format(backward_cls, args.label, pred_score))
    sum_out[0, backward_cls].backward()

@torch.no_grad()
def get_grad_cam_weights(grads):
    weights = {}
    for grad_name in grads:
        _grad = grads[grad_name]['out'][0][0]
        L, C = _grad.size()
        H = W = int(L ** 0.5)
        _grad = _grad.view(H, W, C).permute(2, 0, 1)
        C, H, W = _grad.size()
        weights[grad_name] = _grad.mean(1).mean(1)
        print(weights[grad_name].max())

    return weights

@torch.no_grad()
def plot_grad_cam(features, weights):
    act_maps = {}
    for name in features:
        hs = features[name]['out'][0]
        L, C = hs.size()
        H = W = int(L ** 0.5)
        hs = hs.view(H, W, C).permute(2, 0, 1)
        C, H, W = hs.size()
        w = weights[name]
        w = w.view(-1, 1, 1).repeat(1, H, W)
        weighted_hs = F.relu(w * hs)
        a_map = weighted_hs
        a_map = a_map.sum(0)
        # a_map /= abs(a_map).max()
        act_maps[name] = a_map
    return act_maps

if __name__ == "__main__":

    global module_id_mapper, features, grads
    module_id_mapper, features, grads = {}, {}, {}

    """
    Please add 
    pretrained_path to yaml file.
    """
    # ===== 0. get setting =====
    parser = argparse.ArgumentParser("Visualize SwinT Large")
    parser.add_argument("-pr", "--pretrained_root", type=str, 
        help="contain {pretrained_root}/best.pt, {pretrained_root}/config.yaml")
    parser.add_argument("-img", "--image", type=str)
    parser.add_argument("-sn", "--save_name", type=str)
    parser.add_argument("-lb", "--label", type=int)
    parser.add_argument("-usl", "--use_label", default=False, type=bool)
    parser.add_argument("-sum_t", "--sum_features_type", default="softmax", type=str)
    args = parser.parse_args()
    
    load_yaml(args, args.pretrained_root + "/config.yaml")

    # ===== 1. build model =====
    model = build_model(pretrainewd_path = args.pretrained_root + "/best.pt",
                        img_size = args.data_size, 
                        fpn_size = args.fpn_size, 
                        num_classes = args.num_classes,
                        num_selects = args.num_selects)

    # ===== 2. load image =====
    img_loader = ImgLoader(img_size = args.data_size)
    img, ori_img = img_loader.load(args.image)

    # ===== 3. forward and backward =====
    img = img.unsqueeze(0) # add batch size dimension
    out = model(img)

    cal_backward(args, out, sum_type="softmax")
    
    # ===== 4. check result =====
    grad_weights = get_grad_cam_weights(grads)
    act_maps = plot_grad_cam(features, grad_weights)

    # ===== 5. show =====
    # cv2.imwrite("./vis_imgs/{}_ori.png".format(args.save_name), ori_img)
    sum_act = None
    resize = torchvision.transforms.Resize((args.data_size, args.data_size))
    for name in act_maps:
        layer_name = "layer: {}".format(name)
        _act = act_maps[name]
        _act /= _act.max()
        r_act = resize(_act.unsqueeze(0))
        act_m = _act.numpy() * 255
        act_m = act_m.astype(np.uint8)
        act_m = cv2.resize(act_m, (args.data_size, args.data_size))
        # cv2.namedWindow(layer_name, 0)
        # cv2.imshow(layer_name, act_m)
        if sum_act is None:
            sum_act = r_act
        else:
            sum_act *= r_act
    
    sum_act /= sum_act.max()
    sum_act = torchvision.transforms.functional.adjust_gamma(sum_act, 1.0)
    sum_act = sum_act.numpy()[0]

    # sum_act *= 255
    # sum_act = sum_act.astype(np.uint8)

    plt.cla()
    cdict = get_cdict()
    cmap = matplotlib.colors.LinearSegmentedColormap("jet_revice", cdict)
    plt.imshow(ori_img[:, :, ::-1] / 255)
    plt.imshow(sum_act, alpha=0.5, cmap=cmap) # , alpha=0.5, cmap='jet'
    plt.axis('off')
    plt.savefig("./{}.jpg".format(args.save_name), 
        bbox_inches='tight', pad_inches=0.0, transparent=True)
    plt.show()
    
    # cv2.namedWindow("ori", 0)
    # cv2.imshow("ori", ori_img)
    # cv2.namedWindow("heat", 0)
    # cv2.imshow("heat", sum_act)
    # cv2.waitKey(0)
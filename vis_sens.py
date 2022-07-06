"""
LEGACY:
    View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
    My Youtube Channel: https://www.youtube.com/user/MorvanZhou
    Dependencies:
    torch: 0.4
    matplotlib
    numpy
"""
"""
TODO:
1. Add option to save the best performing model after pruning, with a request for directory to save this file in. File needs to be named logits_best.pkl so that this is compatible with retraining multiple times.
    1 a. Secondary check to ensure certain weights are zeroed out when retraining multiple times and observe what variables get affected and how.

"""
import os
import cv2
import time
import copy
import math
import random
import argparse

import numpy             as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from collections.abc          import Iterable

# Pytorch imports
import torch
import torchvision

import torch.nn             as nn
import torch.optim          as optim
import torch.utils.data     as Data

from torchvision               import datasets, transforms
from torch.autograd            import Variable
from torch.optim.lr_scheduler  import MultiStepLR

# Custom imports 
from utils                     import save_checkpoint, load_checkpoint, accuracy
from data_loader               import data_loader
 
# Model imports
from vgg16.model.vgg16         import VGG16    as vgg
from resnet56.model.resnet56   import RESNET56 as resnet56
from resnet50.model.resnet50   import RESNET50 as resnet50

# Seed setup
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

def terp(min_value, max_value, value):
    return ((value - min_value)/(max_value - min_value))

def gen_mask(prune_percent, upper_prune_limit, parent_key, children_key, clusters, children_clusters, labels_file, labels_children_file, parents_file, device, final_weights, save_children):
    # Check range of validity of pruning amount
    I_parent        = np.load('results/'+parents_file, allow_pickle=True).item()
    labels          = np.load('results/'+labels_file, allow_pickle=True).item()
    labels_children = np.load('results/'+labels_children_file, allow_pickle=True).item()
    
    # Create a copy
    init_weights   = copy.deepcopy(final_weights)
    for key in init_weights.keys():
        init_weights[key] = init_weights[key].detach()
    
    min_value = 1000000
    max_value = 0
    for key in init_weights.keys():
        if np.max(init_weights[key].reshape(-1).cpu().numpy()) > max_value:
            max_value = np.max(init_weights[key].reshape(-1).cpu().numpy())

        if np.min(init_weights[key].reshape(-1).cpu().numpy()) < min_value:
            min_value = np.min(init_weights[key].reshape(-1).cpu().numpy())

    sorted_rho     = None
    mask_weights   = {}

    # Flatten I_parent dictionary
    for looper in range(len(I_parent.keys())):
        if sorted_rho is None:
            sorted_rho = I_parent[str(looper)].reshape(-1)
    
        else:
            sorted_rho =  np.concatenate((sorted_rho, I_parent[str(looper)].reshape(-1)))
    
        # END IF
    
    # END FOR
    
    # Compute unique values
    sorted_rho = np.unique(sorted_rho)
    sorted_rho = np.sort(sorted_rho)
    
    # Compute pruning threshold 
    cutoff_index   = np.round(prune_percent * sorted_rho.shape[0]).astype('int')
    cutoff_value   = sorted_rho[cutoff_index]
    
    for num_layers in range(len(parent_key)):
        parent_k   = parent_key[num_layers]
        children_k = children_key[num_layers]
        shapes     = init_weights[children_k].shape

        # Compute sensitivity scores of children based on grandchildren's weights
        if num_layers < len(parent_key)-1:
            grand_children_key = children_key[num_layers+1]
            mod_weights        = terp(min_value, max_value, init_weights[grand_children_key].detach().cpu().numpy())

            if len(mod_weights.shape)>2:
                mod_weights = np.mean(mod_weights, (2,3))

            importance = mod_weights / np.repeat(np.sum(mod_weights, 1), mod_weights.shape[1]).reshape(mod_weights.shape)  
            reg = []
            stepper = importance.shape[1]/children_clusters[num_layers]

            for looper in range(children_clusters[num_layers]):
                reg.append(np.mean(np.sum(importance[:, int(looper*stepper):int(looper*stepper + stepper)],1)))

            reg = np.asarray(reg)

        else:
            reg = np.ones(children_clusters[num_layers])

        # END IF

        for child in np.argsort(reg)[np.round(save_children[num_layers]*reg.shape[0]).astype('int'):]:
        #for child in range(children_clusters[num_layers]):
            # Pre-compute % of weights to be removed in layer
            layer_remove_per = float(len(np.where(I_parent[str(num_layers)].reshape(-1) <= cutoff_value)[0]) * shapes[0]* shapes[1]/children_clusters[num_layers]/ clusters[num_layers]) / np.prod(shapes[:2])
    
            if layer_remove_per >= upper_prune_limit[num_layers]:
                local_sorted_rho   = np.sort(np.unique(I_parent[str(num_layers)].reshape(-1)))
                cutoff_value_local = local_sorted_rho[np.round(upper_prune_limit[num_layers] * local_sorted_rho.shape[0]).astype('int')]
            
            else:
                cutoff_value_local = cutoff_value
    
            # END IF
    
            try:
                for group_1 in np.where(I_parent[str(num_layers)][child, :] <= cutoff_value_local)[0]:
                    # EDIT: This only works when groups == filters, need to find an alternative solution
                    group_p, group_c = np.meshgrid(np.where(labels[str(num_layers)]==group_1)[0], np.where(labels_children[str(num_layers)]==child)[0])
                    if 'linear' in children_k and 'conv' in parent_k:
                        for group in zip(group_p.reshape(-1), group_c.reshape(-1)):
                            group_pp, group_cc = group
                            init_weights[children_k][group_cc, int(group_pp*init_weights[children_k].shape[1]/clusters[num_layers]):int(group_pp*init_weights[children_k].shape[1]/clusters[num_layers] + init_weights[children_k].shape[1]/clusters[num_layers])] = 0.

                    else:
                        init_weights[children_k][group_c, group_p] = 0.

                # END IF
            except:
                import pdb; pdb.set_trace()

            #for group_1 in range(clusters[num_layers]):
            #    if (I_parent[str(num_layers)][child, group_1] <= cutoff_value_local):
            #        for group_p in np.where(labels[str(num_layers)]==group_1)[0]:
            #            for group_c in np.where(labels_children[str(num_layers)]==child)[0]:
            #                try:
            #                    init_weights[children_k][group_c, group_p] = 0.
            #                except:
            #                    import pdb; pdb.set_trace()
            #    # END IF
    
            # END FOR
    
        # END FOR
 
        mask_weights[children_k] = np.ones(init_weights[children_k].shape)
        mask_weights[children_k][np.where(init_weights[children_k].detach().cpu()==0)] = 0
    
    # END FOR
    
    if len(parent_key) > 1:
        total_count = 0
        valid_count = 0
    
        for num_layers in range(len(parent_key)):
            #total_count = 0
            #valid_count = 0
    
            total_count += init_weights[children_key[num_layers]].reshape(-1).shape[0]
            valid_count += len(np.where(init_weights[children_key[num_layers]].detach().cpu().reshape(-1)!=0.)[0])
            
            #print('Compression percent of layer %s is %f'%(children_key[num_layers], 1 - valid_count/float(total_count)))
    
    else:
        valid_count = len(np.where(init_weights[children_key[0]].detach().cpu().reshape(-1)!= 0.0)[0])
        total_count = float(init_weights[children_key[0]].reshape(-1).shape[0])
    
    
    
    true_prune_percent = valid_count / float(total_count) * 100.
    #import pdb; pdb.set_trace()
    return mask_weights, true_prune_percent, total_count





def train():

    np.random.seed(1993)
    total_acc = []

    # Load Data
    #trainloader, testloader, extraloader, train_data, test_data = data_loader('CIFAR10', 128, 650)

   
    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    model = vgg(num_classes=10).to(device)

    key = 'conv12.weight'
    # Retrain Setup 
    # Load Model prior to Sens 
    init_weights = load_checkpoint('BASELINE_CIFAR10_VGG16_RETRAIN_1/final/logits_15.530962873931625.pkl')
    heat_map_1 = np.mean(init_weights[key].detach().cpu().numpy(), (2,3))
    heat_map_1[heat_map_1<0] = 1
    heat_map_1[heat_map_1>0] = 1
    del init_weights

    init_weights = load_checkpoint('BASELINE_CIFAR10_VGG16_RETRAIN_1/final/logits_14.950434027777778.pkl')
    heat_map_2 = np.mean(init_weights[key].detach().cpu().numpy(), (2,3))
    heat_map_2[heat_map_2<0] = 1
    heat_map_2[heat_map_2>0] = 1

    plt.imshow(heat_map_1, cmap='gray')
    plt.show()
    plt.imshow(heat_map_2, cmap='gray')
    plt.show()
    #f, (ax1, ax2) = plt.subplots(1, 2)

    #AX1 = ax1.imshow(heat_map_1, cmap='gray')
    #AX2 = ax2.imshow(heat_map_2, cmap='gray')


    return 0 

if __name__ == "__main__":

    #parser = argparse.ArgumentParser()

    #parser.add_argument('--Epoch',                type=int   ,   default=10)
    #parser.add_argument('--Batch_size',           type=int   ,   default=128)
    #parser.add_argument('--Lr',                   type=float ,   default=0.001)
    #parser.add_argument('--Dataset',              type=str   ,   default='CIFAR10')
    #parser.add_argument('--Dims',                 type=int   ,   default=10)
    #parser.add_argument('--Expt_rerun',           type=int   ,   default=1)
    #parser.add_argument('--Milestones',           nargs='+',     type=float,       default=[100,150,200])
    #parser.add_argument('--Opt',                  type=str   ,   default='sgd')
    #parser.add_argument('--Weight_decay',         type=float ,   default=0.0001)
    #parser.add_argument('--Model',                type=str   ,   default='resnet32')
    #parser.add_argument('--Gamma',                type=float ,   default=0.1)
    #parser.add_argument('--Nesterov',             action='store_true' , default=False)
    #parser.add_argument('--Device_ids',           nargs='+',     type=int,       default=[0])
    #parser.add_argument('--Retrain',              type=str)
    #parser.add_argument('--Retrain_mask',         type=str)
    #parser.add_argument('--Labels_file',          type=str)
    #parser.add_argument('--Labels_children_file',          type=str)
    #parser.add_argument('--parent_key',           nargs='+',     type=str,       default=['conv1.weight'])
    #parser.add_argument('--children_key',         nargs='+',     type=str,       default=['conv2.weight'])
    #parser.add_argument('--parent_clusters',      nargs='+',     type=int,       default=[8])
    #parser.add_argument('--children_clusters',    nargs='+',     type=int,       default=[8])
    #parser.add_argument('--upper_prune_limit',    nargs='+',     type=float,    default=0.75)
    #parser.add_argument('--save_children',        nargs='+',     type=float,    default=0.75)
    #parser.add_argument('--upper_prune_per',      type=float,    default=0.1)
    #parser.add_argument('--lower_prune_per',      type=float,    default=0.9)
    #parser.add_argument('--prune_per_step',       type=float,    default=0.001)

    #parser.add_argument('--baseline_prune_per',   type=float,    default=16.57)
    #parser.add_argument('--baseline_perf',        type=float,    default=93.43)
    #
    ## Keywords to save best re-trained file
    #parser.add_argument('--Save_dir',             type=str   ,   default='.')

    ## Keyword to parallely run training instances
    #parser.add_argument('--key_id',               type=int)
    #parser.add_argument('--samples_per_class',    type=int,      default=650)
    #
    #args = parser.parse_args()
 
    train()

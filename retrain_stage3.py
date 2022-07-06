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


def gen_mask(prune_percent, upper_prune_limit, parent_key, children_key, clusters, children_clusters, labels_file, labels_children_file, parents_file, device, final_weights):
    # Check range of validity of pruning amount
    I_parent        = np.load('results/'+parents_file, allow_pickle=True).item()
    labels          = np.load('results/'+labels_file, allow_pickle=True).item()
    labels_children = np.load('results/'+labels_children_file, allow_pickle=True).item()
    
    # Create a copy
    init_weights   = copy.deepcopy(final_weights)
    for key in init_weights.keys():
        init_weights[key] = init_weights[key].detach()
    
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
    
        for child in range(children_clusters[num_layers]):
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





def train(Epoch, Batch_size, Lr, Dataset, Dims, Milestones, Rerun, Opt, Weight_decay, Model, Gamma, Nesterov, Device_ids, Retrain, Retrain_mask, Labels_file, Labels_children_file, prune_percent, parent_key, children_key, parent_clusters, children_clusters, upper_prune_limit, baseline_prune_per, samples_per_class):

    print("Experimental Setup: ", args)

    np.random.seed(1993)
    total_acc = []

    # Load Data
    trainloader, testloader, extraloader, train_data, test_data = data_loader(Dataset, Batch_size, samples_per_class)

   
    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    if Model == 'vgg':
        model = vgg(num_classes=Dims).to(device)

    elif Model == 'resnet56':
        model = resnet56(num_classes=Dims).to(device)

    elif Model == 'resnet50':
        model = resnet50(num_classes=Dims).to(device)

    else:
        print('Invalid optimizer selected. Exiting')
        exit(1)

    # END IF

    # Retrain Setup 
    # Load old state
    model.load_state_dict(load_checkpoint(Retrain))


    # EDIT: Modify to include mask generation and pruning
    mask, true_prune_percent, total_count = gen_mask(prune_percent, upper_prune_limit, parent_key, children_key, parent_clusters, children_clusters, Labels_file, Labels_children_file, Retrain_mask, device, load_checkpoint(Retrain))

    if true_prune_percent <= baseline_prune_per:

        print('Requested prune percentage is %f'%(prune_percent))
        print('True pruning percentage is %f'%(true_prune_percent))
        print('Total parameter count is %d'%(total_count))

        # Apply masks
        model.setup_masks(mask)

        logsoftmax = nn.LogSoftmax()

        params     = [p for p in model.parameters() if p.requires_grad]

        if Opt == 'rms':
            optimizer  = optim.RMSprop(model.parameters(), lr=Lr)

        else:
            optimizer  = optim.SGD(params, lr=Lr, momentum=0.9, weight_decay=Weight_decay, nesterov=Nesterov)

        # END IF

        scheduler      = MultiStepLR(optimizer, milestones=Milestones, gamma=Gamma)    
        best_model_acc = 0.0
        best_model     = None

        # Training Loop
        for epoch in range(Epoch):
            # Call every epoch to reset seeds
            if Dataset != 'IMAGENET':
                train_data.set_up_new_seeds()

            running_loss = 0.0

            # Setup Model To Train 
            model.train()

            start_time = time.time()

            for step, data in enumerate(trainloader):
        
                # Extract Data From Loader
                x_input, y_label = data

                ########################### Data Loader + Training ##################################
                one_hot                                       = np.zeros((y_label.shape[0], Dims))
                one_hot[np.arange(y_label.shape[0]), y_label] = 1
                y_label                                       = torch.Tensor(one_hot) 


                if x_input.shape[0]:
                    x_input, y_label = x_input.to(device), y_label.to(device)

                    optimizer.zero_grad()

                    outputs = model(x_input)
                    loss    = torch.mean(torch.sum(-y_label * logsoftmax(outputs), dim=1))

                    loss.backward()
                    optimizer.step()
        
                    ## Add Loss Element
                    if np.isnan(loss.item()):
                        import pdb; pdb.set_trace()

                    # END IF

                # END IF

                ########################### Data Loader + Training ##################################
 
   
            scheduler.step()

            end_time = time.time()
 
            epoch_acc = 100*accuracy(model, testloader, device)


            if best_model_acc < epoch_acc:
                best_model_acc = epoch_acc
                best_model     = copy.deepcopy(model)
                save_checkpoint(epoch, 0, model, optimizer, args.Save_dir+'/0/logits_'+str(true_prune_percent)+'.pkl')
        
        # END FOR

        print('Requested prune percentage is %f'%(prune_percent))
        print('Highest accuracy for true pruning percentage %f is %f'%(true_prune_percent, best_model_acc))
        print('Total number of parameters is %d\n'%(total_count))

    else:
        true_prune_percent = 100.
        best_model_acc     = 0.
        
        best_model = {}
        optimizer  = {}

    # END IF

    return true_prune_percent, best_model_acc, best_model, optimizer  

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--Epoch',                type=int   ,   default=10)
    parser.add_argument('--Batch_size',           type=int   ,   default=128)
    parser.add_argument('--Lr',                   type=float ,   default=0.001)
    parser.add_argument('--Dataset',              type=str   ,   default='CIFAR10')
    parser.add_argument('--Dims',                 type=int   ,   default=10)
    parser.add_argument('--Expt_rerun',           type=int   ,   default=1)
    parser.add_argument('--Milestones',           nargs='+',     type=float,       default=[100,150,200])
    parser.add_argument('--Opt',                  type=str   ,   default='sgd')
    parser.add_argument('--Weight_decay',         type=float ,   default=0.0001)
    parser.add_argument('--Model',                type=str   ,   default='resnet32')
    parser.add_argument('--Gamma',                type=float ,   default=0.1)
    parser.add_argument('--Nesterov',             action='store_true' , default=False)
    parser.add_argument('--Device_ids',           nargs='+',     type=int,       default=[0])
    parser.add_argument('--Retrain',              type=str)
    parser.add_argument('--Retrain_mask',         type=str)
    parser.add_argument('--Labels_file',          type=str)
    parser.add_argument('--Labels_children_file',          type=str)
    parser.add_argument('--parent_key',           nargs='+',     type=str,       default=['conv1.weight'])
    parser.add_argument('--children_key',         nargs='+',     type=str,       default=['conv2.weight'])
    parser.add_argument('--parent_clusters',      nargs='+',     type=int,       default=[8])
    parser.add_argument('--children_clusters',    nargs='+',     type=int,       default=[8])
    parser.add_argument('--upper_prune_limit',    nargs='+',     type=float,    default=0.75)
    parser.add_argument('--upper_prune_per',      type=float,    default=0.1)
    parser.add_argument('--lower_prune_per',      type=float,    default=0.9)
    parser.add_argument('--prune_per_step',       type=float,    default=0.001)

    parser.add_argument('--baseline_prune_per',   type=float,    default=16.57)
    parser.add_argument('--baseline_perf',        type=float,    default=93.43)
    
    # Keywords to save best re-trained file
    parser.add_argument('--Save_dir',             type=str   ,   default='.')

    # Keyword to parallely run training instances
    parser.add_argument('--key_id',               type=int)
    parser.add_argument('--samples_per_class',    type=int,      default=650)
    
    args = parser.parse_args()
 
    possible_prune_percents   = np.arange(args.lower_prune_per, args.upper_prune_per, step=args.prune_per_step)

    best_prune_percent      = args.baseline_prune_per
    best_pp_model_acc       = args.baseline_perf

    true_prune_percent, best_model_acc, model, optimizer = train(args.Epoch, args.Batch_size, args.Lr, args.Dataset, args.Dims, args.Milestones, args.Expt_rerun, args.Opt, args.Weight_decay, args.Model, args.Gamma, args.Nesterov, args.Device_ids, args.Retrain, args.Retrain_mask, args.Labels_file, args.Labels_children_file, possible_prune_percents[args.key_id-1], args.parent_key, args.children_key, args.parent_clusters, args.children_clusters, args.upper_prune_limit, args.baseline_prune_per, args.samples_per_class)

    if true_prune_percent < best_prune_percent and best_model_acc >= best_pp_model_acc:
        print('Saving best model: True prune percent %f, Best Acc. %f'%(true_prune_percent, best_model_acc))
        save_checkpoint(args.Epoch, 0, model, optimizer, args.Save_dir+'/0/logits_'+str(true_prune_percent)+'.pkl')
            
        

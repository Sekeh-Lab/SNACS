import time
import copy
import torch
import random
import argparse
import multiprocessing
import sklearn.metrics

import numpy             as np
import torch.nn          as nn
import matplotlib.pyplot as plt

from tqdm            import tqdm
from sklearn.svm     import SVC
from sklearn.cluster import KMeans
from hungarian       import Hungarian

# Custom Imports
from data_loader             import data_loader
from utils                   import activations, sub_sample_uniform, mi, mi_edge
from utils                   import save_checkpoint, load_checkpoint, accuracy, mi, mi_edge

# Model imports
from vgg16.model.vgg16         import VGG16 as vgg
from resnet56.model.resnet56   import RESNET56 as resnet56 
from resnet50.model.resnet50   import RESNET50 as resnet50 

# Fixed Backend To Force Dataloader To Be Consistent
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
est = 'edge'



def confusion_matrix_st(preds, labels):
        num_inst   = len(preds)
        num_labels = np.max(labels) + 1 

        conf_matrix = np.zeros( (num_labels, num_labels))

        for i in range(0, num_inst):
            gt_i = labels[i]
            pr_i = preds[i]
            conf_matrix[gt_i, pr_i] = conf_matrix[gt_i, pr_i] + 1 

        return conf_matrix

def collect_activations(model, extraloader, device, parent_key, children_key):
    print("----------------------------------")
    print("Collecting activations from layers")

    p1_op = {}
    c1_op = {}

    unique_keys = np.unique(np.union1d(parent_key, children_key)).tolist()
    act         = {}
    lab         = {}

    for item_key in unique_keys:
        act[item_key], lab[item_key] = activations(extraloader, model, device, item_key)

    return act, lab

# Compute RHO'
def compute_rho_tick(I_weighted_parents, parent_key, children_key, I_parent):
    for num_layers in range(len(parent_key)):
        parent_k = parent_key[num_layers]
        child_k  = children_key[num_layers]
    
        I_weighted_parents[str(num_layers)] = I_parent[str(num_layers)]/np.repeat(np.sum(I_parent[str(num_layers)], 1), (I_parent[str(num_layers)].shape[1])).reshape(I_parent[str(num_layers)].shape)

# EDIT: NEED TO VERIFY WORKING
def compute_importance(current_c_k, weights_dir, name_postfix, parent_key, children_key):
    I_parent        = np.load('results/'+weights_dir+'I_parent_'+name_postfix+'.npy', allow_pickle=True).item()
    labels          = np.load('results/'+weights_dir+'Labels_'+name_postfix+'.npy', allow_pickle=True).item()
    labels_children = np.load('results/'+weights_dir+'Labels_children_'+name_postfix+'.npy', allow_pickle=True).item()

    #parent_key   = ['conv1.weight', 'conv2.weight', 'conv3.weight']
    #children_key = ['conv2.weight', 'conv3.weight', 'linear1.weight']

    # Compute rho' 
    I_weighted_parents = {}
    importance         = {}
    return_importance  = {}
    compute_rho_tick(I_weighted_parents, parent_key, children_key, I_parent)

    # Compute importance of children 
    for num_layers in np.arange(len(parent_key)-1, -1, -1):
        if num_layers == len(parent_key)-1:
            importance[str(num_layers)] = np.repeat(1./I_parent[str(num_layers)].shape[0], I_parent[str(num_layers)].shape[0])

        else:
            try:
                importance[str(num_layers)] = np.sum(np.multiply(I_weighted_parents[str(num_layers+1)], np.repeat(importance[str(num_layers+1)], I_weighted_parents[str(num_layers+1)].shape[1]).reshape(I_weighted_parents[str(num_layers+1)].shape)), 0)
            except:
                import pdb; pdb.set_trace()
        if current_c_k == children_key[num_layers]:
            return_importance['0'] = importance[str(num_layers)]

    return return_importance


def prune_layer(init_I_parent, prune_percent, parent_key, children_key, init_weights, labels, labels_children, clusters, clusters_children, save_children, weights_dir, name_postfix, parents, children):
    mask_weights       = {}
    I_weighted_parents = {}

    # EDIT: NEED TO VERIFY WORKING
    compute_rho_tick(I_weighted_parents, parent_key, children_key, init_I_parent)

    parent_k   = parent_key[0]
    children_k = children_key[0]
    importance = compute_importance(children_k, weights_dir, name_postfix, parents, children)

    # Compute unique values, sort and obtain cutoff thresholds
    sorted_weights = init_I_parent['0'].reshape(-1)
    sorted_weights = np.unique(sorted_weights)
    sorted_weights = np.sort(sorted_weights)
    cutoff_index   = np.round(prune_percent * sorted_weights.shape[0]).astype('int')
    cutoff_value   = sorted_weights[cutoff_index]
   
    # EDIT: NEED TO VERIFY WORKING
    for child in np.argsort(importance['0'])[::-1][np.round(save_children*importance['0'].shape[0]).astype('int'):]:
    #for child in range(clusters_children[0]):
        try:
            for group_1 in np.where(init_I_parent['0'][child, :] <= cutoff_value)[0]:
                # EDIT: This only works when groups == filters, need to find an alternative solution
                group_p, group_c = np.meshgrid(np.where(labels['0']==group_1)[0], np.where(labels_children['0']==child)[0])
                if 'linear' in children_k and 'conv' in parent_k:
                    for group in zip(group_p.reshape(-1), group_c.reshape(-1)):
                        group_p, group_c = group
                        init_weights[children_k][group_c, int(group_p*init_weights[children_k].shape[1]/clusters[0]):int(group_p*init_weights[children_k].shape[1]/clusters[0] + init_weights[children_k].shape[1]/clusters[0])] = 0.

                else:
                    init_weights[children_k][group_c, group_p] = 0.

                # END IF
        except:
            import pdb; pdb.set_trace()

        # END FOR

    # END FOR

    mask_weights[children_k] = np.ones(init_weights[children_k].shape)
    mask_weights[children_k][np.where(init_weights[children_k].detach().cpu()==0)] = 0

    # Calculate Statistics of compression for layers
    total_count = 0
    valid_count = 0

    for num_layers in range(len(parent_key)):
        layer_total_count = 0
        layer_valid_count = 0

        layer_total_count += init_weights[children_key[num_layers]].reshape(-1).shape[0]
        layer_valid_count += len(np.where(init_weights[children_key[num_layers]].detach().cpu().reshape(-1)!=0.)[0])
    
        # Number of parents fully removed
        if len(init_weights[children_key[num_layers]].shape) > 2:
            no_parents_removed = init_weights[children_key[num_layers]].shape[1] - np.count_nonzero(np.sum(np.mean(abs(init_weights[children_key[num_layers]].detach().cpu().numpy()), (2,3)), 0))

        else:
            no_parents_removed = init_weights[children_key[num_layers]].shape[1] - np.count_nonzero(np.sum(abs(init_weights[children_key[num_layers]].detach().cpu().numpy()), 0))

        # END IF

        total_count += layer_total_count 
        valid_count += layer_valid_count 

        print('Layer %s: Compression %f, Parents removed: %d/%d'%(children_key[num_layers], 1 - layer_valid_count/float(layer_total_count), no_parents_removed, init_weights[children_key[num_layers]].shape[1]))

    true_prune_percent = valid_count / float(total_count) * 100.

    return init_weights, true_prune_percent, mask_weights

#### Main Code Executor 
def calc_gamma(model, dataset, parent_key, children_key, clusters, clusters_children, weights_dir, cores, name_postfix, samples_per_class, dims, phi_type, prune_percent, save_children, inits, device_ids, parents, children):

    #### Load Data ####
    trainloader, testloader, extraloader, train_data, test_data = data_loader(dataset, 64, samples_per_class)

    #### Load Model, Weights and Rhos ####
    init_weights   = load_checkpoint(weights_dir+'logits_best.pkl')
    init_I_parent  = np.load('results/'+weights_dir+'I_parent_'+name_postfix+'_'+parent_key[0]+'_'+children_key[0]+'.npy', allow_pickle=True).item()

    for key in init_weights.keys():
        init_weights[key] = init_weights[key].detach().detach()


    #### Compute original activations ####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model == 'vgg':
        model  = vgg(num_classes=dims).to(device)

    if model == 'resnet56':
        model  = resnet56(num_classes=dims).to(device)

    if model == 'resnet50':
        model  = resnet50(num_classes=dims).to(device)

    ## Extend model to multi-gpu setup
    #model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.load_state_dict(init_weights)
    model.eval()

    act, lab = collect_activations(model, extraloader, device, parent_key, children_key)
    act[parent_key[0]][act[parent_key[0]]<0] = 0
    act[children_key[0]][act[children_key[0]]<0] = 0

    labels          = {}
    labels_children = {}

    labels['0']          = np.load('results/'+weights_dir+'Labels_'+name_postfix+'_'+parent_key[0]+'_'+children_key[0]+'.npy', allow_pickle=True).item()['0'] 
    labels_children['0'] = np.load('results/'+weights_dir+'Labels_children_'+name_postfix+'_'+parent_key[0]+'_'+children_key[0]+'.npy', allow_pickle=True).item()['0']

    #### Prune layer to desired percentage ####
    init_weights, true_prune_percent, mask_weights = prune_layer(init_I_parent, prune_percent, parent_key, children_key, init_weights, labels, labels_children, clusters, clusters_children, save_children, weights_dir, name_postfix, parents, children)

    #### Apply pruned weights and recollect activations #### 

    model.load_state_dict(init_weights)
    model.eval()

    new_act, new_lab = collect_activations(model, extraloader, device, parent_key, children_key)
    new_act[parent_key[0]][new_act[parent_key[0]]<0] = 0
    new_act[children_key[0]][new_act[children_key[0]]<0] = 0


    #### Metric 1: Train Kmeans with orig activations and test performance on new activations ####
    #LL          = sklearn.metrics.pairwise.cosine_similarity(act[children_key[0]], new_act[children_key[0]])
    #overall_kmeans_accuracy = np.mean(np.diagonal(LL))
    clf = SVC(random_state=0, tol=1e-5)
    clf.fit(act[children_key[0]], lab[children_key[0]])

    preds = clf.predict(new_act[children_key[0]])
    overall_kmeans_accuracy = np.sum(preds==new_lab[children_key[0]])/float(new_lab[children_key[0]].shape[0])

    return overall_kmeans_accuracy, true_prune_percent

if __name__=='__main__':

    """"
    Sample Input values
    parent_key        = ['conv1.weight','conv2.weight','conv3.weight','conv4.weight','conv5.weight','conv6.weight','conv7.weight','conv8.weight','conv9.weight', 'conv10.weight','conv11.weight','conv12.weight','conv13.weight', 'linear1.weight']
    children_key      = ['conv2.weight','conv3.weight','conv4.weight','conv5.weight','conv6.weight','conv7.weight','conv8.weight','conv9.weight','conv10.weight','conv11.weight','conv12.weight','conv13.weight','linear1.weight', 'linear3.weight']
    alg               = '1a_group'
    clusters          = [8,8,8,8,8,8,8,8,8,8,8,8,8,8]
    clusters_children = [8,8,8,8,8,8,8,8,8,8,8,8,8,8]
    load_weights  = '/z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/logits_best.pkl'
    save_data_dir = '/z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/'
    """


    parser = argparse.ArgumentParser()

    parser.add_argument('--model',                type=str)
    parser.add_argument('--dataset',              type=str)
    parser.add_argument('--weights_dir',          type=str)
    parser.add_argument('--cores',                type=int)
    parser.add_argument('--dims',                 type=int, default=10)
    parser.add_argument('--key_id',               type=int)
    parser.add_argument('--samples_per_class',    type=int,      default=250)
    parser.add_argument('--parent_clusters',      nargs='+',     type=int,       default=[8])
    parser.add_argument('--children_clusters',    nargs='+',     type=int,       default=[8])
    parser.add_argument('--name_postfix',         type=str)
    parser.add_argument('--phi_type',             type=str)
    parser.add_argument('--prune_percent',        type=float)
    parser.add_argument('--inits',                type=int)
    parser.add_argument('--device_ids',           nargs='+',     type=int,       default=[0])

    args = parser.parse_args()

    print('Selected key id is %d'%(args.key_id))

    if args.model == 'vgg':
        layer_names = np.load('vgg16_dict.npy', allow_pickle=True).item()

    elif args.model == 'resnet56':
        layer_names = np.load('resnet56_dict.npy', allow_pickle=True).item()
        if args.key_id <= 20:
            args.parent_clusters[0] = 16 
        if args.key_id <= 19:
            args.children_clusters[0] = 16 
        if args.key_id > 20 and args.key_id <= 38:
            args.parent_clusters[0] = 32 
        if args.key_id > 19 and args.key_id <= 37:
            args.children_clusters[0] = 32 
        if args.key_id > 38:
            args.parent_clusters[0] = 64
        if args.key_id > 37: 
            args.children_clusters[0] = 64 

    elif args.model == 'resnet50':
        layer_names = np.load('resnet50_dict.npy', allow_pickle=True).item()

    else:
        print('Invalid model selected. Exiting!')
        exit(1)

    # END IF

    parents     = layer_names['parents']
    children    = layer_names['children']

    if args.key_id == len(parents):
        args.children_clusters = [args.dims]

    kmeans_perf = np.zeros((np.arange(0.0, 1.0, 0.1).shape[0], np.arange(1,99,2).shape[0])) 
    prune_perf  = np.zeros((np.arange(0.0, 1.0, 0.1).shape[0], np.arange(1,99,2).shape[0])) 

    idx_i = -1 
    idx_j = -1 

    for save_children in [0.0]:#np.arange(0.0, 1.00, 0.1):
        idx_i += 1
        idx_j = -1 
        for prune_percent in np.arange(1, 99, 2): 
            idx_j += 1
            overall_kmeans_accuracy,  true_prune_percent = calc_gamma(args.model, args.dataset, [parents[args.key_id-1]], [children[args.key_id-1]], args.parent_clusters, args.children_clusters, args.weights_dir, args.cores, args.name_postfix, args.samples_per_class, args.dims, args.phi_type, prune_percent/100., save_children, args.inits, args.device_ids, parents, children)

            kmeans_perf[idx_i, idx_j] = overall_kmeans_accuracy
            prune_perf[idx_i, idx_j]  = true_prune_percent

        # Save Data wrt. metric, percent children saved
        np.savetxt('csv_files/'+args.model+'_'+args.name_postfix+'_svm_'+parents[args.key_id-1]+'_'+children[args.key_id-1]+'_'+str(save_children*100.)+'SENS_RHO.csv',kmeans_perf[idx_i],delimiter=',')
        np.savetxt('csv_files/'+args.model+'_'+args.name_postfix+'_prune_'+parents[args.key_id-1]+'_'+children[args.key_id-1]+'_'+str(save_children*100.)+'SENS_RHO.csv',prune_perf[idx_i],delimiter=',')

    for idx_i in range(kmeans_perf.shape[0]):
        ul_compression = np.arange(1,99,2.)[np.where(kmeans_perf[idx_i] >= np.max(kmeans_perf[idx_i]) - 0.7*(np.max(kmeans_perf[idx_i]) - np.min(kmeans_perf[idx_i])))[0][-1]]
        print('UL on compression for layer %s, saving %f children is %f'%(children[args.key_id-1], np.arange(0.0, 1.00, 0.1)[idx_i]*100., ul_compression))

    print(kmeans_perf[0])
    print('Code Execution Complete')

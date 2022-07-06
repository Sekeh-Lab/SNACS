import time
import copy
import torch
import random
import argparse
import multiprocessing

import numpy             as np

# Custom imports
from data_loader             import data_loader
from utils                   import save_checkpoint, load_checkpoint, accuracy, activations, mi, mi_edge

# Model imports
from vgg16.model.vgg16         import VGG16    as vgg
from resnet56.model.resnet56   import RESNET56 as resnet56
from resnet50.model.resnet50   import RESNET50 as resnet50

### Fixed Backend To Force Dataloader To Be Consistent ###
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def idxs(labels, samples_per_class, dataset):
    unq_lab = np.unique(labels)
    op_idxs = None
    for lab in unq_lab:
        if op_idxs is None:
            op_idxs = np.random.choice(np.where(labels == lab)[0], samples_per_class)
        else:
            op_idxs = np.concatenate((op_idxs, np.random.choice(np.where(labels == lab)[0], samples_per_class)))

    # Verify
    for idx in np.arange(0, op_idxs.shape[0], samples_per_class):
        assert(np.unique(labels[op_idxs[idx:idx+samples_per_class]]).shape[0] == 1)

    np.save(dataset+'_idxs_'+str(samples_per_class)+'.npy', op_idxs)

def terp(min_value, max_value, value):
    return ((value - min_value)/(max_value - min_value))

#### Conditional Mutual Information Computation ####
def cmi(data):
    clusters, c1_op, child, p1_op, num_layers, labels, labels_children, phi_type, est_type = data 
    I_value = np.zeros((clusters,))

    for group_1 in range(clusters):
        if est_type == 'mst':
            I_value[group_1] += mi(X=c1_op[str(num_layers)][:, np.where(labels_children[str(num_layers)]==child)[0]], Y=p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], Z=p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]!=group_1)[0]]) 

        else:
            I_value[group_1] += mi_edge(X=c1_op[str(num_layers)][:, np.where(labels_children[str(num_layers)]==child)[0]], Y=p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], Z=p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]!=group_1)[0]], phi_type=phi_type) 

    # END FOR 


    return I_value 

#### CMI group-based computation ####
def cmi_group(nlayers, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children, cores, phi_type, init_weights, children_key, max_value, min_value, est_type):

    print("----------------------------------")
    print("Begin Execution of Group-based CMI estimation")
    print("Estimator Type: %s, Phi Type: %s"%(est_type, phi_type))

    pool = multiprocessing.Pool(cores)

    for num_layers in range(nlayers):
        data = []

        for child in range(clusters_children[num_layers]):
            data.append([clusters[num_layers], c1_op, child, p1_op, num_layers, labels, labels_children, phi_type, est_type])

        # END FOR 
        
        data = tuple(data)
        ret_values = pool.map(cmi, data)

        for child in range(clusters_children[num_layers]):

            if 'weight' in phi_type:
                network_weights = terp(min_value, max_value, init_weights[children_key[num_layers]][child, :].detach().cpu().numpy())
                assert(np.max(network_weights) <= 1 and np.min(network_weights)>=0)

                if 'conv' in children_key[num_layers]:
                    network_weights = np.mean(network_weights, (1,2))

                mean_step_size = int(network_weights.shape[0]/clusters[num_layers])

                for elements in  np.arange(0,network_weights.shape[0], mean_step_size):
                    if phi_type == 'weight' or phi_type == 'activation_weight':
                        I_parent[str(num_layers)][child, int(elements/mean_step_size)] = np.multiply(np.mean(network_weights[elements:elements + mean_step_size]), ret_values[child][int(elements/mean_step_size)])

                    elif phi_type == 'exp_weight' or phi_type == 'exp_activation_weight':
                        I_parent[str(num_layers)][child, int(elements/mean_step_size)] = np.multiply(np.exp(-np.mean(network_weights[elements:elements + mean_step_size])**2/2.), ret_values[child][int(elements/mean_step_size)])

                    elif phi_type == 'weight_sq' or phi_type == 'activation_weight_sq':
                        I_parent[str(num_layers)][child, int(elements/mean_step_size)] = np.multiply(np.mean(network_weights[elements:elements + mean_step_size])**2, ret_values[child][int(elements/mean_step_size)])

            else:
                I_parent[str(num_layers)][child,:] = ret_values[child]

        # END FOR 

    # END FOR


#### Main Code Executor 
def calc_cmi(model, dataset, parent_key, children_key, clusters, clusters_children, weights_dir, cores, name_postfix, samples_per_class, dims, phi_type, est_type, device_ids):


    #### Load Model ####
    init_weights   = load_checkpoint(weights_dir+'logits_best.pkl')

    #### Compute the min and max range of weight values ####
    min_value = 1000000
    max_value = 0
    for key in init_weights.keys():
        if np.max(init_weights[key].detach().reshape(-1).cpu().numpy()) > max_value:
            max_value = np.max(init_weights[key].detach().reshape(-1).cpu().numpy())

        if np.min(init_weights[key].detach().reshape(-1).cpu().numpy()) < min_value:
            min_value = np.min(init_weights[key].detach().reshape(-1).cpu().numpy())


    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #### Load Network ####
    if model == 'vgg':
        model = vgg(num_classes=dims).to(device)

    elif model == 'resnet56':
        model = resnet56(num_classes=dims).to(device)

    elif model == 'resnet50':
        model = resnet50(num_classes=dims).to(device)

    else:
        print('Invalid model selected')

    # END IF

    ## Extend model to multi-gpu setup
    #model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.load_state_dict(init_weights)
    model.eval()

    #### Load Data ####
    trainloader, testloader, extraloader, train_data, test_data = data_loader(dataset, 64, samples_per_class)
 
    
    # Original Accuracy 
    acc = 100.*accuracy(model, testloader, device)
    print('Accuracy of the original network: %f %%\n' %(acc))

    nlayers         = 1 
    labels          = {}
    labels_children = {}
    children        = {}
    I_parent        = {}

    # Obtain Activations
    print("----------------------------------")
    print("Collecting activations from layers")

    act_start_time = time.time()
    p1_op = {}
    c1_op = {}

    unique_keys = np.unique(np.union1d(parent_key, children_key)).tolist()
    act         = {}
    lab         = {}

    # Since we collect activations from extraloader, we already use only a small subset of training data
    for item_key in unique_keys:
        act[item_key], lab[item_key] = activations(extraloader, model, device, item_key)

    for item_idx in range(len(parent_key)):
        p1_op[str(item_idx)] = copy.deepcopy(act[parent_key[item_idx]]) 
        c1_op[str(item_idx)] = copy.deepcopy(act[children_key[item_idx]])

    act_end_time   = time.time()

    print("Time taken to collect subset of activations is : %f seconds\n"%(act_end_time - act_start_time))

    #idxs(lab['conv2.weight'], samples_per_class, 'IMAGENET')

    del act, lab

    labels[str(0)]          = np.zeros((init_weights[children_key[0]].shape[1],))
    labels_children[str(0)] = np.zeros((init_weights[children_key[0]].shape[0],))
    I_parent[str(0)]        = np.zeros((clusters_children[0], clusters[0]))

    #### Compute Clusters/Groups ####
    print("----------------------------------")
    print("Begin Clustering Layers: %s and %s\n"%(parent_key[0], children_key[0]))

    # Parents
    if p1_op[str(0)].shape[1] == clusters[0]:
        labels[str(0)] = np.arange(clusters[0])

    else:
        if labels[str(0)].shape[0]%clusters[0] > 0:
            print('Cluster assignments are not balanced. Please retry with a different cluster size.')
            exit(1)
        else:
            labels[str(0)] = np.repeat(np.arange(clusters[0]), labels[str(0)].shape[0]/clusters[0])

    # END IF

    # Children
    if c1_op[str(0)].shape[1] == clusters_children[0]:
        labels_children[str(0)] = np.arange(clusters_children[0])

    else:
        if labels_children[str(0)].shape[0]%clusters_children[0] > 0:
            print('Cluster assignments are not balanced. Please retry with a different cluster size.')
            exit(1)
        else: 
            labels_children[str(0)] = np.repeat(np.arange(clusters_children[0]), labels_children[str(0)].shape[0]/clusters_children[0])

    # END IF

    ### Compute CMI ####
    cmi_start_time = time.time()
    cmi_group(nlayers, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children, cores, phi_type, init_weights, children_key, max_value, min_value, est_type)
    cmi_end_time = time.time()

    print('Time taken to compute CMI: %s, Phi: %s is %f'%(est_type, phi_type, cmi_end_time - cmi_start_time))

    #### Save results ####
    np.save('results/'+weights_dir+'I_parent_'+name_postfix+'.npy', I_parent)
    np.save('results/'+weights_dir+'Labels_'+name_postfix+'.npy', labels)
    np.save('results/'+weights_dir+'Labels_children_'+name_postfix+'.npy', labels_children)

    return cmi_end_time - cmi_start_time

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
    parser.add_argument('--est_type',             type=str, default='edge')
    parser.add_argument('--device_ids',           nargs='+',     type=int,       default=[0])
    parser.add_argument('--expt_rerun',           type=int, default=1)

    args = parser.parse_args()

    print('Selected key id is %d'%(args.key_id))


    if args.model == 'vgg':
        layer_names = np.load('vgg16_dict.npy', allow_pickle=True).item()
        parents     = layer_names['parents']
        children    = layer_names['children']

    elif args.model == 'resnet56':
        layer_names = np.load('resnet56_dict.npy', allow_pickle=True).item()
        parents     = layer_names['parents']
        children    = layer_names['children']

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
        parents     = layer_names['parents']
        children    = layer_names['children']


    if args.key_id == len(parents):
        args.children_clusters = [args.dims]

    cmi_time = []

    for rerun in range(args.expt_rerun): 
        cmi_time.append(calc_cmi(args.model, args.dataset, [parents[args.key_id-1]], [children[args.key_id-1]], args.parent_clusters, args.children_clusters, args.weights_dir, args.cores, args.name_postfix +'_'+parents[args.key_id-1]+'_'+children[args.key_id-1], args.samples_per_class, args.dims, args.phi_type, args.est_type, args.device_ids))

    print('Average CMI runtime over %d trials is %f'%(args.expt_rerun, np.mean(cmi_time)))
    print('Std. of CMI runtime over %d trials is %f'%(args.expt_rerun, np.std(cmi_time)))
    print('CMI run times ', np.array(cmi_time))

    print('Code Execution Complete')

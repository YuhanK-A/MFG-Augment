import torch
import numpy as np
from utils.start_iteration import start_train
from utils_new import divide_feature_vector_set_into_two_sets
import argparse
import pprint as pp
import sys
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt


# ================================================
#                      Main
# ================================================

# ===============
# Initializations
# ===============

torch_seed = np.random.randint(low=-sys.maxsize - 1, high=sys.maxsize,dtype=np.int64)
torch.random.manual_seed(torch_seed)
np_seed = np.random.randint(low=0, high=2 ** 32 - 1,dtype=np.int64)
np.random.seed(np_seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
if device == torch.device('cpu'):
    print('NOTE: USING ONLY THE CPU')

#我的
# Two_sets = GetSamplesFromTwoSets(train_dataset, 5)
# _, Initial_set,_ = Two_sets.construct_set_of_the_same_label()
# _, _, Terminal_set = Two_sets.construct_set_of_the_same_label()
#我的new
set_of_feature_vectors_for_all = torch.load('./Feature_CIFAR10_Autoencoder/feature_set_for_all.pth')['feature_set_for_all']
set_of_feature_vectors_for_all.requires_grad = False
Initial_set, Terminal_set = divide_feature_vector_set_into_two_sets(set_of_feature_vectors_for_all)

# ==============================
# Hyperparameter Argument Parser
# ==============================

# TODO: Move this to another file.



parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--torch_seed', default=torch_seed)
parser.add_argument('--np_seed', default=np_seed)
parser.add_argument('--max_epochs', default=int(6e5))
parser.add_argument('--TT', default=1.0,                                       help='Final time')
parser.add_argument('--device', default=device)
parser.add_argument('--print_rate', default=1000)
parser.add_argument('--dim', default=768)
parser.add_argument('--nu', default=0)


parser.add_argument('--ham_func', default='hom1')
parser.add_argument('--ham_scale', default=8)


parser.add_argument('--lam_disc_00', default=1)
parser.add_argument('--lam_gen_TT', default=0)
parser.add_argument('--lam_ham', default=1)
parser.add_argument('--lam_G', default=1)


parser.add_argument('--lam_obstacle', default=5)
parser.add_argument('--lam_congestion', default=0)
parser.add_argument('--lam_quadrun', default=0)


parser.add_argument('--batch_size', default=50)
parser.add_argument('--ns', default=100,                                       help='Network size')
parser.add_argument('--disc_lr', default=4e-4)
parser.add_argument('--gen_lr', default=1e-4)
parser.add_argument('--sgd_momentum', default=0.0,                             help='SGD only')
parser.add_argument('--betas', default=(0.5, 0.9),                           help='Adam only')
parser.add_argument('--weight_decay', default=1e-4)
parser.add_argument('--act_func_disc', default='tanh',                         help='Activation function for discriminator')
parser.add_argument('--act_func_gen', default='relu',                          help='Activation function for generator')
parser.add_argument('--gen_every_disc', default=1,                             help='How many discriminator updates before one generator update')
parser.add_argument('--h', default=0.5,                                        help='ResNet step-size')
parser.add_argument('--lr_milestones', default=[])
parser.add_argument('--hjb_error', default=1,                                help='initial value for L2 hjb error')
parser.add_argument('--hjb_error_milestones', default=[])
parser.add_argument('--grad_norm_clip_value', default=np.inf)

bool_logging = True
parser.add_argument('--do_logging', default=bool_logging)
parser.add_argument('--do_compute_validation', default=bool_logging)
parser.add_argument('--num_val_points', default=64*64)  # 原来的default=64*64
parser.add_argument('--num_plot_samples', default=100)
parser.add_argument('--show_plots', default=False)
parser.add_argument('--plot_window', default=3)
parser.add_argument('--Initial_set', default=Initial_set)
parser.add_argument('--Terminal_set', default=Terminal_set)

args = vars(parser.parse_args())
args['experiment_name'] = '_dim-' + str(args['dim']) \
                          + '__cong_bn' \
                          # + '__nu-' + str(args['nu']) \
                          # + '__hamscale-' + str(args['ham_scale']) \
                          # + '__lam_congestion_obstacle-' + str(args['lam_congestion']) + '-' + str(args['lam_obstacle']) \
                          # + '__hjb_error-' + str(args['hjb_error']) \
                          # + '__act_disc-' + args['act_func_disc'] \
                          # + '__act_gen-' + args['act_func_gen'] \
                          # + '__disc_gen_lr-' + str(args['disc_lr']) + '_' + str(args['gen_lr']) \
                          # + '__betas-' + str(args['betas'][0]) + '_' + str(args['betas'][1]) \
                          # + '__bs-' + str(args['batch_size']) \
                          # + '__psi_dim_1-2_L1__scale-1'

                          # + '__disc_ham_loss2-L2' \
                          # + '__rho0_var-0.1' \
                          # + '__wider_obst'
                          # + '__rho0var-0.2' \
                          # + '__gen_every_disc-' + str(args['gen_every_disc'])
                          # + '__ns-' + str(args['ns'])
                          # + '__grad_norm_clip-' + str(args['grad_norm_clip_value']) \
                          # + '__only_grad_clip_gen' \
                          # + '__lam_disc_00-' + str(args['lam_disc_00']) \
                          # + '__nonresnet'


# ===========================
# Start hyperparameter search
# ===========================

pp.pprint(args)
the_logger = start_train(args)





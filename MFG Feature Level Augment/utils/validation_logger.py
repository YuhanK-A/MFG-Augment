import torch
from utils.utils import DISC_STRING, GEN_STRING
from utils.helpers_train_once import get_hjb_error, get_interaction_error
from os.path import join
import numpy as np
import csv
import matplotlib.pyplot as plt


# =========
# Setup CSV
# =========

def _setup_validation_csv(csv_filepath):
    """
    Setup the csv file, mainly the columns we want
    """
    with open(csv_filepath, 'w') as f:
        csv_columns = ['epoch', 'disc_hjb_error', 'gen_hjb_error', 'disc_obstacle_loss_val',
                       'disc_congestion_loss_val', 'disc_quadrun_loss_val', 'gen_obstacle_loss_val',
                       'gen_congestion_loss_val', 'gen_quadrun_loss_val']
        f.write(','.join(csv_columns) + '\n')

# =======================
# Validation logger class
# =======================

class ValidationLogger(object):

    def __init__(self, args, validation_dict=None):
        self.do_compute_validation = args['do_compute_validation']
        if self.do_compute_validation:
            assert(validation_dict is not None)
            self.tt_val = validation_dict['tt_sampler'](args['num_val_points'], args['device'])
            self.zz_val = validation_dict['zz_sampler'](args['Initial_set'], args['num_val_points'], args['dim']).to(args['device'])
            self.zz2_val = validation_dict['zz_sampler'](args['Terminal_set'], args['num_val_points'], args['dim']).to(args['device'])
            self.lam_G = args['lam_G']
            self.lam_obstacle = args['lam_obstacle']
            self.lam_congestion = args['lam_congestion']
            self.lam_quadrun = args['lam_quadrun']
            self.num_val_points = args['num_val_points']
            self.dim = args['dim']
            self.experiment_dir = validation_dict['experiment_dir']
            self.print_rate = args['print_rate']
            self.nu = args['nu']
            self.samples_from_Terminal_set = validation_dict['samples_from_Terminal_set'](args['num_val_points'], args['Terminal_set']).to(args['device'])

            # Things to log
            self.validation_disc_hjb_error = []  # discriminator hjb error on value after updating DISCRIMINATOR
            self.validation_gen_hjb_error = []  # discriminator hjb error on value after updating GENERATOR
            self.validation_disc_obstacle_loss = []
            self.validation_gen_obstacle_loss = []
            self.validation_disc_congestion_loss = []
            self.validation_gen_congestion_loss = []
            self.validation_disc_quadrun_loss = []
            self.validation_gen_quadrun_loss = []

            # Ones tensor of size phi out
            self.ones_of_size_phi_out_val = torch.ones(args['num_val_points'] * args['dim'], 1).to(args['device']) \
                                            if args['nu'] > 0 else torch.ones(args['num_val_points'], 1).to(args['device'])
            # Grad outputs vec for laplacian
            list_1 = []
            for i in range(args['dim']):
                vec = torch.zeros(size=(args['num_val_points'], args['dim']), dtype=torch.float).to(args['device'])
                vec[:, i] = torch.ones(size=(args['num_val_points'],)).to(args['device'])
                list_1.append(vec)
            self.grad_outputs_vec_val = torch.cat(list_1, dim=0)

            # Make csv
            self.csv_filepath = join(self.experiment_dir, 'validation_log.csv')
            _setup_validation_csv(self.csv_filepath)

    def get_hjb_error_on_validation(self, train_dict, discriminator, generator):
        if self.do_compute_validation:
            discriminator.eval()
            generator.eval()

            rhott_samples_val = generator(self.tt_val, self.zz_val, self.samples_from_Terminal_set).detach().requires_grad_(True)

            # hjb_error_tensor, some_info_hjb \
            #     = val_logger_get_hjb_error(train_dict, self.tt_val, rhott_samples_val, self.num_val_points,
            #                                self.ones_of_size_phi_out_val, self.grad_outputs_vec_val, DISC_STRING)
            hjb_error_tensor, some_info_hjb = \
                get_hjb_error(train_dict, self.tt_val, rhott_samples_val, self.num_val_points,
                              self.ones_of_size_phi_out_val, self.grad_outputs_vec_val, DISC_STRING)
            temp_assert = self.dim if self.nu > 0 else 1
            assert(hjb_error_tensor.size() == torch.Size([self.num_val_points*temp_assert, 1]))
            hjb_error_tensor = hjb_error_tensor[:self.num_val_points]
            assert (hjb_error_tensor.size() == torch.Size([self.num_val_points, 1]))

            G_total_tensor, some_info_interaction \
                = get_interaction_error(train_dict, self.tt_val, self.zz2_val, rhott_samples_val,
                                        self.num_val_points, DISC_STRING)
            assert(G_total_tensor.size() == torch.Size([self.num_val_points, 1]))

            hjb_error_val_tensor = torch.norm(hjb_error_tensor + self.lam_G * G_total_tensor, dim=1).mean(dim=0)

            hjb_error_val = hjb_error_val_tensor.item()

            # This frees up the graph TODO: (this del is unnecessary because graph automatically freed, delete in future)
            del hjb_error_val_tensor
            del hjb_error_tensor
            del G_total_tensor
            # del some_info_interaction
            # del some_info_hjb

            discriminator.train()
            generator.train()

            return hjb_error_val, some_info_hjb, some_info_interaction

    def log_validation(self, train_dict, discriminator, generator, disc_or_gen):
        if self.do_compute_validation:
            hjb_error_val, some_info_hjb, some_info_interaction = self.get_hjb_error_on_validation(train_dict, discriminator, generator)

            # Log hjb error and interaction error
            if disc_or_gen == DISC_STRING:
                self.validation_disc_hjb_error.append(hjb_error_val)
                # Log interaction
                if some_info_interaction['G_obstacle_loss'] is not 'N/A':
                    self.validation_disc_obstacle_loss.append(some_info_interaction['G_obstacle_loss'])
                if some_info_interaction['G_congestion_loss'] is not 'N/A':
                    self.validation_disc_congestion_loss.append(some_info_interaction['G_congestion_loss'])
                if some_info_interaction['G_quadrun_loss'] is not 'N/A':
                    self.validation_disc_quadrun_loss.append(some_info_interaction['G_quadrun_loss'])
            elif disc_or_gen == GEN_STRING:
                self.validation_gen_hjb_error.append(hjb_error_val)
                # Log interaction
                if some_info_interaction['G_obstacle_loss'] is not 'N/A':
                    self.validation_gen_obstacle_loss.append(some_info_interaction['G_obstacle_loss'])
                if some_info_interaction['G_congestion_loss'] is not 'N/A':
                    self.validation_gen_congestion_loss.append(some_info_interaction['G_congestion_loss'])
                if some_info_interaction['G_quadrun_loss'] is not 'N/A':
                    self.validation_gen_quadrun_loss.append(some_info_interaction['G_quadrun_loss'])

            else:
                raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')

    def write_validation_csv(self, epoch):
        if self.do_compute_validation:
            num_data = 1  # self.print_rate
            mean_disc_hjb_error_val = np.mean(self.validation_disc_hjb_error[-num_data:])
            mean_gen_hjb_error_val = np.mean(self.validation_gen_hjb_error[-num_data:])

            # Interaction errors
            mean_disc_obstacle_loss_val = np.mean(self.validation_disc_obstacle_loss[-num_data:]) if self.lam_obstacle > 0 else np.nan
            mean_disc_congestion_loss_val = np.mean(self.validation_disc_congestion_loss[-num_data:]) if self.lam_congestion > 0 else np.nan
            mean_disc_quadrun_loss_val = np.mean(self.validation_disc_quadrun_loss[-num_data:]) if self.lam_quadrun > 0 else np.nan

            mean_gen_obstacle_loss_val = np.mean(self.validation_gen_obstacle_loss[-num_data:]) if self.lam_obstacle > 0 else np.nan
            mean_gen_congestion_loss_val = np.mean(self.validation_gen_congestion_loss[-num_data:]) if self.lam_congestion > 0 else np.nan
            mean_gen_quadrun_loss_val = np.mean(self.validation_gen_quadrun_loss[-num_data:]) if self.lam_quadrun > 0 else np.nan

            with open(self.csv_filepath, 'a') as f:
                the_string = f'{epoch},{mean_disc_hjb_error_val},{mean_gen_hjb_error_val},' \
                             f'{mean_disc_obstacle_loss_val},{mean_disc_congestion_loss_val},{mean_disc_quadrun_loss_val},' \
                             f'{mean_gen_obstacle_loss_val},{mean_gen_congestion_loss_val},{mean_gen_quadrun_loss_val}\n'
                f.write(the_string)

    def plot_validation_curve(self):
        if self.do_compute_validation:
            validation_log_csv_filepath = join(self.experiment_dir, 'validation_log.csv')
            with open(validation_log_csv_filepath, newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
            data = data[1:]
            data_np = np.array(data, dtype=np.float)
            fig, ax = plt.subplots(1)
            ax.plot(data_np[:, 0], np.log(data_np[:, 1]))
            plt.title('Discriminator HJB residual | epoch vs. log(hjb_error)')
            fig.savefig(join(self.experiment_dir, 'discriminator_hjb_error_plot.png'))
            plt.close(fig)
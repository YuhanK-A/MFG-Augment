import torch
import numpy as np
import datetime
import os
from os.path import join
import pprint as pp
from utils.validation_logger import ValidationLogger
from utils.utils import DISC_STRING, GEN_STRING
import pickle


# =====================
# Some helper functions
# =====================

def _get_time_string():
    """
    Just get the current time in a string
    """
    ## New format, but messes with legacy
    # out = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]
    # out = out[:10] + '__' + out[11:]  # separate year-month-day from hour-minute-seconds

    # Old format
    out = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]

    return out

def _save_files(mypath, dest):
    """
    Save the run files so we can rerun in the future,
    and we know what settings we use.
    """
    from os import listdir
    from os.path import isfile, join
    import shutil
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print('onlyfiles:', onlyfiles)
    if not os.path.exists(dest):
        os.makedirs(dest)
    for file in onlyfiles:
        shutil.copy2(join(mypath, file), dest)

def _setup_training_csv(csv_filepath):
    """
    Setup the csv file, mainly the columns we want
    """
    # Setup training log csv
    with open(csv_filepath, 'w') as f:
        csv_columns = ['epoch', 'disc_loss', 'disc_hjb_error', 'gen_loss', 'gen_hjb_error', 'disc_obstacle_loss',
                       'disc_congestion_loss', 'disc_quadrun_loss', 'gen_obstacle_loss', 'gen_congestion_loss',
                       'gen_quadrun_loss']
        f.write(','.join(csv_columns) + '\n')


# =====================
# Logging when training
# =====================

def do_logging(the_logger, some_info_training, disc_or_gen):
    if disc_or_gen == DISC_STRING:
        log_dict = {'disc_loss': some_info_training['total_loss'],
                    'disc_hjb_error': some_info_training['ham_loss2'],
                    'disc_obstacle_loss': some_info_training['G_obstacle_loss'],
                    'disc_congestion_loss': some_info_training['G_congestion_loss'],
                    'disc_quadrun_loss': some_info_training['G_quadrun_loss']}
        the_logger.log_training(log_dict)
    elif disc_or_gen == GEN_STRING:
        log_dict = {'gen_loss': some_info_training['total_loss'],
                    'gen_hjb_error': some_info_training['ham_loss2'],
                    'gen_obstacle_loss': some_info_training['G_obstacle_loss'],
                    'gen_congestion_loss': some_info_training['G_congestion_loss'],
                    'gen_quadrun_loss': some_info_training['G_quadrun_loss']}
        the_logger.log_training(log_dict)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')


# ============
# Logger class
# ============

class Logger(object):
    """
    A logger to log things.
    """
    def __init__(self, args, validation_dict=None):
        self.do_logging = args['do_logging']
        if self.do_logging:
            # Make the experiment directory
            if 'test' in args and args['test'] == True:  # For testing models only
                run_dir = 'TestRun_' + _get_time_string() + '_' + args['experiment_name']
                self.experiment_dir = join('Tests', run_dir)
            else:
                run_dir = 'Run_' + _get_time_string() + '_' + args['experiment_name']
                self.experiment_dir = join('Experiments', run_dir)

            if not os.path.exists(self.experiment_dir):
                os.makedirs(self.experiment_dir)
            _save_files('./', self.experiment_dir)
            _save_files('./utils', join(self.experiment_dir, 'utils'))

            # Make plots folder
            self.plots_dir = join(self.experiment_dir, 'plots')
            if not os.path.exists(self.plots_dir):
                os.makedirs(self.plots_dir)

            # Figure out which interactions are active
            self.lam_obstacle = args['lam_obstacle']
            self.lam_congestion = args['lam_congestion']
            self.lam_quadrun = args['lam_quadrun']

            # Make lists for things we want to log
            self.training_disc_loss = []
            self.training_disc_hjb_error = []
            self.training_gen_loss = []
            self.training_gen_hjb_error = []
            self.training_disc_obstacle_loss = []
            self.training_disc_congestion_loss = []
            self.training_disc_quadrun_loss = []
            self.training_gen_obstacle_loss = []
            self.training_gen_congestion_loss = []
            self.training_gen_quadrun_loss = []

            # Make CSV file for logging
            self.csv_filepath = join(self.experiment_dir, 'train_log.csv')
            _setup_training_csv(self.csv_filepath)

            # Print rate
            self.print_rate = args['print_rate']

            # Print/save hyperparameters
            with open(join(self.experiment_dir, 'experiment_args.txt'), 'w') as f:
                pp.pprint(args, f)
            with open(join(self.experiment_dir, 'experiment_args.pkl'), 'wb') as f:
                pickle.dump(args, f)

            # Validation logger
            validation_dict['experiment_dir'] = self.experiment_dir
            self.validation_logger = ValidationLogger(args, validation_dict)

        else: # Don't do logging
            self.validation_logger = ValidationLogger(args)


    def log_training(self, the_dict):
        """
        Append stuff to the list for logging
        """
        if self.do_logging:
            if 'disc_loss' in the_dict:
                self.training_disc_loss.append(the_dict['disc_loss'])
            if 'disc_hjb_error' in the_dict:
                self.training_disc_hjb_error.append(the_dict['disc_hjb_error'])
            if 'gen_loss' in the_dict:
                self.training_gen_loss.append(the_dict['gen_loss'])
            if 'gen_hjb_error' in the_dict:
                self.training_gen_hjb_error.append(the_dict['gen_hjb_error'])
            if 'disc_obstacle_loss' in the_dict:
                self.training_disc_obstacle_loss.append(the_dict['disc_obstacle_loss'])
            if 'disc_congestion_loss' in the_dict:
                self.training_disc_congestion_loss.append(the_dict['disc_congestion_loss'])
            if 'disc_quadrun_loss' in the_dict:
                self.training_disc_quadrun_loss.append(the_dict['disc_quadrun_loss'])
            if 'gen_obstacle_loss' in the_dict:
                self.training_gen_obstacle_loss.append(the_dict['gen_obstacle_loss'])
            if 'gen_congestion_loss' in the_dict:
                self.training_gen_congestion_loss.append(the_dict['gen_congestion_loss'])
            if 'gen_quadrun_loss' in the_dict:
                self.training_gen_quadrun_loss.append(the_dict['gen_quadrun_loss'])


    def write_training_csv(self, epoch):
        """
        Write to the csv file
        """
        if self.do_logging:
            mean_disc_loss = np.mean(self.training_disc_loss[-self.print_rate:])
            mean_disc_hjb_error = np.mean(self.training_disc_hjb_error[-self.print_rate:])
            mean_gen_loss = np.mean(self.training_gen_loss[-self.print_rate:])
            mean_gen_hjb_error = np.mean(self.training_gen_hjb_error[-self.print_rate:])

            # Interaction terms
            mean_disc_obstacle_loss = np.mean(self.training_disc_obstacle_loss[-self.print_rate:]) if self.lam_obstacle > 0 else 'N/A'
            mean_disc_congestion_loss = np.mean(self.training_disc_congestion_loss[-self.print_rate:]) if self.lam_congestion > 0 else 'N/A'
            mean_disc_quadrun_loss = np.mean(self.training_disc_quadrun_loss[-self.print_rate:]) if self.lam_quadrun > 0 else 'N/A'
            mean_gen_obstacle_loss = np.mean(self.training_gen_obstacle_loss[-self.print_rate:]) if self.lam_obstacle > 0 else 'N/A'
            mean_gen_congestion_loss = np.mean(self.training_gen_congestion_loss[-self.print_rate:]) if self.lam_congestion > 0 else 'N/A'
            mean_gen_quadrun_loss = np.mean(self.training_gen_quadrun_loss[-self.print_rate:]) if self.lam_quadrun > 0 else 'N/A'

            with open(self.csv_filepath, 'a') as f:
                the_string = f'{epoch},{mean_disc_loss},{mean_disc_hjb_error},{mean_gen_loss},{mean_gen_hjb_error},' \
                             f'{mean_disc_obstacle_loss},{mean_disc_congestion_loss},{mean_disc_quadrun_loss},' \
                             f'{mean_gen_obstacle_loss},{mean_gen_congestion_loss},{mean_gen_quadrun_loss}\n'
                f.write(the_string)

    def save_nets(self, the_dict):
        """
        Save the discriminator and generator models,
        and their optimizers
        """
        if self.do_logging:
            self.model_path = join(self.experiment_dir, 'models')
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)

            epoch = the_dict['epoch']
            discriminator = the_dict['discriminator']
            discriminator_optimizer = the_dict['discriminator_optimizer']
            generator = the_dict['generator']
            generator_optimizer = the_dict['generator_optimizer']

            # Save discriminator
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'optimizer': discriminator_optimizer.state_dict()},
            join(self.model_path, f'discriminator-epoch-{epoch}.pth.tar'))

            # Save generator
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer': generator_optimizer.state_dict(),
                'gen_mu': generator.mu,
                'gen_std': generator.std},
                join(self.model_path, f'generator-epoch-{epoch}.pth.tar'))

    def save_plot(self, epoch, fig):
        """
        Save plots.
        """
        if self.do_logging:
            fig.savefig(join(self.plots_dir, f'plot-{epoch}.png'))
            fig.savefig(join(self.experiment_dir, f'latest_plot.png'))

    def print_to_console(self, some_info_training, disc_or_gen):
        """
        Stuff to print to the console.
        """
        with torch.no_grad():
            # Setup variables
            ham_loss = some_info_training['ham_loss']
            ham_loss2 = some_info_training['ham_loss2']
            ham_loss_max = some_info_training['ham_loss_max']
            error_msg = f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}'
            G_obstacle_loss = some_info_training['G_obstacle_loss']
            G_quadratic_run_loss = some_info_training['G_quadrun_loss']
            G_congestion_loss = some_info_training['G_congestion_loss']
            phi_lap_xx = some_info_training['phi_lap_xx']
            total_loss = some_info_training['total_loss']
            if disc_or_gen == DISC_STRING:
                disc_00_loss = some_info_training['disc_00_loss']
            elif disc_or_gen == GEN_STRING:
                pass  # gen_TT_loss = some_info_training['gen_TT_loss']
            else:
                raise ValueError(error_msg)

            # Start printing to the console
            if disc_or_gen == DISC_STRING:
                print('DISCRIMINATOR losses:')
            elif disc_or_gen == GEN_STRING:
                print('GENERATOR losses:')
            else:
                raise ValueError(error_msg)
            print('ham_loss:', ham_loss)
            print('ham_loss2:', ham_loss2)
            print('ham_loss_max:', ham_loss_max)
            if disc_or_gen == DISC_STRING:
                print('disc_00_loss:', disc_00_loss)
            elif disc_or_gen == GEN_STRING:
                pass  # print('gen_TT_loss:', gen_TT_loss)
            print('G_obstacle_loss:', G_obstacle_loss)
            print('G_congestion_loss:', G_congestion_loss)
            print('G_quadrun_loss:', G_quadratic_run_loss)
            print('phi_lap_xx:', phi_lap_xx)
            print('total_loss:', total_loss)

            print()

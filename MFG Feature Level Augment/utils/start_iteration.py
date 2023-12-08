from utils.utils import act_funcs, MultiStepScheduler, uniform_time_sampler, make_obstacle_contour_plot, DISC_STRING, \
    GEN_STRING, make_plots
from func_defs import sample_rho0, ham_funcs, sample_rho0_fast_cuda
from neural_net_defs import *
from utils.train_once import train_once
from utils.logger import Logger, do_logging
from utils.func_get_two_sets import samples_from_Terminal_set
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np




def start_train(args):
    # =======================
    # Set the hyperparameters
    # =======================

    max_epochs = args['max_epochs']
    TT = args['TT']
    device = args['device']
    print_rate = args['print_rate']
    dim = args['dim']
    nu = args['nu']
    ham_func = ham_funcs[args['ham_func']]
    ham_scale = args['ham_scale']

    hjb_error_d = args['hjb_error']
    hjb_error_g = hjb_error_d
    lam_disc_00 = args['lam_disc_00']
    lam_gen_TT = args['lam_gen_TT']
    lam_ham = args['lam_ham']
    lam_obstacle = args['lam_obstacle']
    lam_congestion = args['lam_congestion']
    lam_quadrun = args['lam_quadrun']
    lam_G = args['lam_G']

    batch_size = args['batch_size']
    ns = args['ns']
    disc_lr = args['disc_lr']
    gen_lr = args['gen_lr']
    weight_decay = args['weight_decay']
    sgd_momentum = args['sgd_momentum']
    betas = args['betas']
    act_func_gen = act_funcs[args['act_func_gen']]
    act_func_disc = act_funcs[args['act_func_disc']]
    h = args['h']
    gen_every_disc = args['gen_every_disc']
    grad_norm_clip_value = args['grad_norm_clip_value']

    lr_milestones = args['lr_milestones']
    hjb_error_milestones = args['hjb_error_milestones']
    Initial_set = args['Initial_set']
    Terminal_set = args['Terminal_set']


    # Ones tensor of size phi out
    ones_of_size_phi_out = torch.ones(batch_size*dim, 1).to(device) if nu > 0 else torch.ones(batch_size, 1).to(device)
    # Grad outputs vec for laplacian
    list_1 = []
    for i in range(dim):
        vec = torch.zeros(size=(batch_size, dim), dtype=torch.float).to(device)
        vec[:, i] = torch.ones(size=(batch_size,)).to(device)
        list_1.append(vec)
    grad_outputs_vec = torch.cat(list_1, dim=0)

    if args['do_compute_validation']:
        assert(args['do_logging'] == args['do_compute_validation'])


    ##我的




    # ==================
    # Setup the learning
    # ==================

    # Compute the mean and variance of rho0 (assuming rho0 is a simple Gaussian)
    temp_sample = sample_rho0(Initial_set, int(1e4), dim).to(device)
    mu = temp_sample.mean(axis=0)
    std = torch.sqrt(temp_sample.var(axis=0))
    print(f'mu: {mu}\nstd: {std}')
    # 可恢复if 0 in std:
    # 可恢复    raise ValueError("std of sample_rho0 has a zero!")
        # std[2:] = torch.ones_like(std[2:])

    # print('mu:', mu)
    # print('std:', std)

    # Get the networks
    discriminator = DiscNet(dim=dim, ns=ns, act_func=act_func_disc, h=h,
                            device=device, TT=TT).to(device)
    generator = GenNet(dim=dim, ns=ns, act_func=act_func_gen, h=h,
                       device=device, mu=mu, std=std, TT=TT).to(device)

    # Weight initializations # TODO: Make this optional in the argparser
    for param in discriminator.parameters():
        if len(param.data.size()) > 1:
            torch.nn.init.kaiming_uniform_(param.data)
    for param in generator.parameters():
        if len(param.data.size()) > 1:
            torch.nn.init.kaiming_uniform_(param.data)

    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_lr, weight_decay=weight_decay, betas=betas)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=gen_lr, weight_decay=weight_decay, betas=betas)

    # disc_optimizer = torch.optim.SGD(discriminator.parameters(), lr=disc_lr, weight_decay=weight_decay, momentum=sgd_momentum)
    # gen_optimizer = torch.optim.SGD(generator.parameters(), lr=gen_lr, weight_decay=weight_decay, momentum=sgd_momentum)

    # LR Scheduler
    disc_optim_sched = torch.optim.lr_scheduler.MultiStepLR(disc_optimizer, lr_milestones, gamma=0.1)
    gen_optim_sched = torch.optim.lr_scheduler.MultiStepLR(gen_optimizer, lr_milestones, gamma=0.1)

    # HJB L2 error scheduler
    hjb_error_d_sched = MultiStepScheduler(hjb_error_d, hjb_error_milestones, gamma=10)
    hjb_error_g_sched = MultiStepScheduler(hjb_error_g, hjb_error_milestones, gamma=10)


    # ======
    # Logger
    # ======
    validation_dict = {'tt_sampler': uniform_time_sampler, 'zz_sampler': sample_rho0, 'samples_from_Terminal_set': samples_from_Terminal_set}
    the_logger = Logger(args, validation_dict)


    # ===============
    # Start iteration
    # ===============
    tic = time.perf_counter()

    # Some constants
    zero = torch.tensor([0], dtype=torch.float).expand((batch_size, 1)).to(device)
    TT = torch.tensor([TT], dtype=torch.float).expand((batch_size, 1)).to(device)

    # Sampler for rho0
    # zz_sampler = lambda: sample_rho0_fast_cuda(batch_size, dim, var_scale=1/10) if torch.cuda.is_available() \
    #     else sample_rho0(batch_size, dim).to(device)
    def zz_sampler(): return sample_rho0(Initial_set, batch_size, dim, var_scale=1/10).to(device)

    # Saved points to plot
    num_plot_samples = args['num_plot_samples']
    zz_saved = sample_rho0(Initial_set, num_plot_samples, dim, var_scale=1/10).to(device)
    zzz_saved = samples_from_Terminal_set(num_plot_samples, Terminal_set).to(device)

    # Make plot of obstacle
    obstacle_plot = make_obstacle_contour_plot(args)

    # Start the iteration
    for epoch in range(max_epochs+1):

        # =========
        # Info dump
        # =========

        if epoch % print_rate == 0:
            print()
            print('-' * 10)
            print(f'epoch: {epoch}\n')
            # print('disc_optimizer lr:', disc_optimizer.state_dict()['param_groups'][0]['lr'])
            # print('gen_optimizer lr:', gen_optimizer.state_dict()['param_groups'][0]['lr'])
            # print('hjb_error_d:', hjb_error_d)
            # print('hjb_error_g:', hjb_error_g)
            # print()

            if epoch != 0:
                # Saving neural network and saving to csv
                the_logger.save_nets({'epoch': epoch,
                                      'discriminator': discriminator,
                                      'discriminator_optimizer': disc_optimizer,
                                      'generator': generator,
                                      'generator_optimizer': gen_optimizer})
                the_logger.write_training_csv(epoch)

                # Validation logger
                the_logger.validation_logger.write_validation_csv(epoch)
                the_logger.validation_logger.plot_validation_curve()

                toc = time.perf_counter()
                print('toc - tic:', toc - tic, '\n')
                tic = time.perf_counter()


        # =========================
        # Setup training dictionary
        # =========================
        train_dict = {'discriminator': discriminator,
                      'generator': generator,
                      'disc_optimizer': disc_optimizer,
                      'gen_optimizer': gen_optimizer,
                      'disc_optim_sched': disc_optim_sched,
                      'gen_optim_sched': gen_optim_sched,
                      'dim': dim,
                      'nu': nu,
                      'ham_func': ham_func,
                      'ham_scale': ham_scale,
                      'epoch': epoch,
                      'print_rate': print_rate,
                      'batch_size': batch_size,
                      'device': device,
                      'zz_sampler': zz_sampler,
                      'zero': zero,
                      'TT': TT,
                      'ones_of_size_phi_out': ones_of_size_phi_out,
                      'grad_outputs_vec': grad_outputs_vec,
                      'lam_disc_00': lam_disc_00,
                      'lam_gen_TT': lam_gen_TT,
                      'lam_ham': lam_ham,
                      'lam_obstacle': lam_obstacle,
                      'lam_congestion': lam_congestion,
                      'lam_quadrun': lam_quadrun,
                      'lam_G': lam_G,
                      'hjb_error_d': hjb_error_d,
                      'hjb_error_g': hjb_error_g,
                      'grad_norm_clip_value': grad_norm_clip_value,
                      'Initial_set': Initial_set,
                      'samples_from_Terminal_set': samples_from_Terminal_set,
                      'Terminal_set': Terminal_set,
                      'the_logger': the_logger}


        # =======================
        # Train phi/discriminator
        # =======================

        if epoch % 1 == 0:  # How many times to update discriminator per generator update.
            some_info_training = train_once(train_dict, DISC_STRING)

        disc_optim_sched.step()
        hjb_error_d = hjb_error_d_sched.step()
        do_logging(the_logger, some_info_training, DISC_STRING)
        if epoch % print_rate == 0:
            the_logger.print_to_console(some_info_training, DISC_STRING)
            the_logger.validation_logger.log_validation(train_dict, discriminator, generator, DISC_STRING)

        # ===================
        # Train rho/generator
        # ===================

        if epoch % gen_every_disc == 0:  # How many times do you want to update discriminator per one update of the generator.
            some_info_training = train_once(train_dict, GEN_STRING)

        gen_optim_sched.step()
        hjb_error_g = hjb_error_g_sched.step()
        do_logging(the_logger, some_info_training, GEN_STRING)
        if epoch % print_rate == 0:
            the_logger.print_to_console(some_info_training, GEN_STRING)
            the_logger.validation_logger.log_validation(train_dict, discriminator, generator, GEN_STRING)

        # ====================
        # Show images and etc.
        # ====================

        if epoch % print_rate == 0:
            plot_dict = {'epoch': epoch, 'print_rate': print_rate, 'num_plot_samples': num_plot_samples,
                         'zz_saved': zz_saved, 'TT': TT, 'device': device, 'zz_sampler': zz_sampler,
                         'obstacle_plot': obstacle_plot, 'show_plots': args['show_plots'],
                         'plot_window': args['plot_window'],'zzz_saved': zzz_saved}
            make_plots(generator, discriminator, plot_dict, the_logger)



    return the_logger

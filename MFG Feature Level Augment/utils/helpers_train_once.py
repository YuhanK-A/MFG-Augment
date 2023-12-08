import torch
from func_defs import sample_rho0, sample_rho0_fast_cuda, psi_func_torch, G_func_obstacle_torch, G_congestion_torch4
from utils.utils import uniform_time_sampler, get_laplacian2, DISC_STRING, GEN_STRING

# ================
# Helper functions
# ================

def get_generator_samples(train_dict, disc_or_gen):
    """
    Get generator samples
    """
    generator = train_dict['generator']
    batch_size = train_dict['batch_size']
    device = train_dict['device']
    zz_sampler = train_dict['zz_sampler']
    lam_congestion = train_dict['lam_congestion']
    TT = train_dict['TT']
    Terminal_set = train_dict['Terminal_set']
    samples_50_from_Terminal_set= train_dict['samples_from_Terminal_set'](batch_size, Terminal_set).to(device)

    zz = zz_sampler()
    tt_samples = uniform_time_sampler(batch_size, device) * TT[0].item()
    # tt_samples = torch.rand(size=(batch_size, 1)).to(device) * TT[0].item()

    if disc_or_gen == 'disc':
        rhott_samples = generator(tt_samples, zz, samples_50_from_Terminal_set).detach().requires_grad_(True)
    elif disc_or_gen == 'gen':
        rhott_samples = generator(tt_samples, zz, samples_50_from_Terminal_set)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be \'disc\' or \'gen\' but got: {disc_or_gen}')

    zz2 = zz_sampler() if lam_congestion > 0 else torch.empty(zz.size()).to(device)

    return zz, zz2, tt_samples, rhott_samples



def get_hjb_error(train_dict, tt_samples, rhott_samples, batch_size, ones_of_size_phi_out, grad_outputs_vec,
                  disc_or_gen):
    """
    Compute the HJB error
    """
    discriminator = train_dict['discriminator']
    dim = train_dict['dim']
    ham_func = train_dict['ham_func']
    ham_scale = train_dict['ham_scale']
    device = train_dict['device']
    TT = train_dict['TT']
    nu = train_dict['nu']

    # Integral for the Hamilton-Jacobi part
    if nu > 0:  # NOTE: Have to repeat if we want to parallelize computing the Laplacian for each sample of the batch.
        rhott_samples = rhott_samples.repeat(repeats=(dim, 1))
        tt_samples = tt_samples.repeat(repeats=(dim, 1))
    tt_samples.requires_grad_(True)  # WARNING: Keep this after generator evaluation, or else you chain rule generator's time variable
    phi_out = discriminator(tt_samples, rhott_samples)
    phi_grad_tt = torch.autograd.grad(outputs=phi_out, inputs=tt_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    phi_grad_xx = torch.autograd.grad(outputs=phi_out, inputs=rhott_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    if nu > 0.0:
        phi_lap_xx = get_laplacian2(phi_grad_xx, rhott_samples, batch_size, dim, grad_outputs_vec, device=device)
        # if train_dict['epoch'] % train_dict['print_rate'] == 0:
            # print('torch.max(phi_lap_xx):', torch.max(phi_lap_xx))
            # print('torch.min(phi_lap_xx):', torch.min(phi_lap_xx))
    else:
        phi_lap_xx = torch.zeros(phi_grad_tt.size()).to(device)

    ham = ham_scale * ham_func(tt_samples, rhott_samples, phi_grad_xx)

    out = TT[0].item() * (phi_grad_tt + nu * phi_lap_xx - ham)
    # print('phi_grad_tt.size():', phi_grad_tt.size())
    # print('phi_grad_xx.size():', phi_grad_xx.size())
    # print('phi_lap_xx.size():', phi_lap_xx.size())
    # print('ham.size():', ham.size())

    # Compute some info
    some_info = {'phi_lap_xx': phi_lap_xx.mean(dim=0).item()}

    return out, some_info



def get_interaction_error(train_dict, tt_samples, zz2, rhott_samples, batch_size, disc_or_gen):
    """
    Computes the interaction terms
    """
    generator = train_dict['generator']
    dim = train_dict['dim']
    device = train_dict['device']
    TT = train_dict['TT']
    lam_obstacle = train_dict['lam_obstacle']
    lam_congestion = train_dict['lam_congestion']
    lam_quadrun = train_dict['lam_quadrun']

    G_total_tensor = torch.zeros((batch_size, 1)).to(device)
    some_info = {'G_obstacle_loss': 'N/A','G_congestion_loss': 'N/A', 'G_quadrun_loss': 'N/A'}

    ### Obstacle ###
    if lam_obstacle > 0:
        G_obstacle_tensor = TT[0].item() * G_func_obstacle_torch(rhott_samples, device)
        G_total_tensor += lam_obstacle * G_obstacle_tensor
        some_info['G_obstacle_loss'] = G_obstacle_tensor.mean(dim=0).item()

    ### Congestion ###
    if lam_congestion > 0:
        G_congestion_tensor = TT[0].item() * G_congestion_torch4(
                                                generator, tt_samples, zz2, rhott_samples, disc_or_gen, 2)
        G_total_tensor += lam_congestion * G_congestion_tensor
        # assert(G_congestion_tensor.size() == torch.Size([batch_size*dim, 1])), "Wrong size."
        some_info['G_congestion_loss'] = G_congestion_tensor.mean(dim=0).item()

    ### Quadratic running cost ###
    if lam_quadrun > 0:
        G_quadratic_tensor = TT[0].item() * psi_func_torch(rhott_samples, device)
        G_total_tensor += lam_quadrun * G_quadratic_tensor
        some_info['G_quadrun_loss'] = G_quadratic_tensor.mean(dim=0).item()

    return G_total_tensor, some_info



# =======================
# Simple helper functions
# =======================

def get_disc_00_loss(discriminator, zero, batch_size, dim, device, Initial_set):
    # Integral of phi_0 * rho_0
    rho0_samples = sample_rho0(Initial_set, batch_size, dim).to(device)
    disc_00_loss = discriminator(zero, rho0_samples).mean(dim=0)

    return disc_00_loss


def get_gen_TT_loss(generator, zz, TT, lam_gen_TT, device):
    """
    Supposed to return the generator loss at time TT.
    But this is currently not supposed to be used, only as a regularizer,
    much like the L2 norm of the HJB error.
    """
    if lam_gen_TT > 0:
        # Integral of (psi - phi) * rho_T (but modified as regularizer)
        rhoTT_samples = generator(TT, zz)
        psi_vals = psi_func_torch(rhoTT_samples, device)
        gen_TT_loss = (psi_vals).mean(dim=0)
    else:
        gen_TT_loss = torch.tensor([0]).to(device)

    return gen_TT_loss


def disc_gen_requires_grads(discriminator, generator, disc_or_gen):
    """
    Turn on requires_grad for the one we're training, and turn off for the one we aren't. For speed.
    """
    if disc_or_gen == DISC_STRING:
        for param in discriminator.parameters():
            param.requires_grad_(True)
        for param in generator.parameters():
            param.requires_grad_(False)
    elif disc_or_gen == GEN_STRING:
        for param in discriminator.parameters():
            param.requires_grad_(False)
        for param in generator.parameters():
            param.requires_grad_(True)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')


def disc_gen_zero_grad(disc_optimizer, gen_optimizer, disc_or_gen):
    """
    Zero the gradients for the one we're training.
    """
    if disc_or_gen == DISC_STRING:
        disc_optimizer.zero_grad()
    elif disc_or_gen == GEN_STRING:
        gen_optimizer.zero_grad()
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')


def disc_gen_grad_clip(discriminator, generator, clip_value, disc_or_gen):
    """
    Clips theg gradient of the discriminator and/or generator
    """
    if disc_or_gen == DISC_STRING:
        pass
        # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_value)
    elif disc_or_gen == GEN_STRING:
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_value)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')


def disc_gen_optimizer_step(disc_optimizer, gen_optimizer, disc_or_gen):
    """
    Take a step of the discriminator or generator optmizers.
    """
    if disc_or_gen == DISC_STRING:
        disc_optimizer.step()
    elif disc_or_gen == GEN_STRING:
        gen_optimizer.step()
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')

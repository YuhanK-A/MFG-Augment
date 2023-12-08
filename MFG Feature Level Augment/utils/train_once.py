from utils.func_get_two_sets import samples_from_Terminal_set
import torch
from utils.utils import DISC_STRING, GEN_STRING, sqeuc
from utils.helpers_train_once import disc_gen_requires_grads, disc_gen_zero_grad, get_generator_samples, get_hjb_error, \
    get_interaction_error, get_disc_00_loss, disc_gen_optimizer_step, get_gen_TT_loss, disc_gen_grad_clip


# ========================
# The main trainer
# ========================

def train_once(train_dict, disc_or_gen):
    """
    Trains the discriminator.
    """
    discriminator = train_dict['discriminator']
    generator = train_dict['generator']
    disc_optimizer = train_dict['disc_optimizer']
    gen_optimizer = train_dict['gen_optimizer']
    dim = train_dict['dim']
    epoch = train_dict['epoch']
    print_rate = train_dict['print_rate']
    batch_size = train_dict['batch_size']
    device = train_dict['device']
    zero = train_dict['zero']
    TT = train_dict['TT']
    lam_disc_00 = train_dict['lam_disc_00']
    lam_gen_TT = train_dict['lam_gen_TT']
    lam_ham = train_dict['lam_ham']
    lam_G = train_dict['lam_G']
    lam_ham2_d = train_dict['hjb_error_d']
    grad_norm_clip_value = train_dict['grad_norm_clip_value']
    nu = train_dict['nu']
    ones_of_size_phi_out = train_dict['ones_of_size_phi_out']
    grad_outputs_vec = train_dict['grad_outputs_vec']
    Initial_set = train_dict['Initial_set']
    samples_from_Terminal_set = train_dict['samples_from_Terminal_set']
    Terminal_set = train_dict['Terminal_set']
    zero_size = batch_size * dim if nu > 0 else batch_size
    error_msg = f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}'
    assert (disc_or_gen == DISC_STRING or disc_or_gen == GEN_STRING), error_msg

    # Activate computing computational graph of discriminator
    disc_gen_requires_grads(discriminator, generator, disc_or_gen)

    # Zero the gradients
    disc_gen_zero_grad(disc_optimizer, gen_optimizer, disc_or_gen)

    # Integral for the Hamilton-Jacobi part
    zz, zz2, tt_samples, rhott_samples = get_generator_samples(train_dict, disc_or_gen)
    hjb_error_tensor, some_hjb_error_info = get_hjb_error(train_dict, tt_samples, rhott_samples, batch_size,
                                                          ones_of_size_phi_out, grad_outputs_vec, disc_or_gen)
    hjb_error_tensor = hjb_error_tensor[:batch_size]
    ham_loss = hjb_error_tensor.mean(dim=0)

    # Interaction terms
    G_total_tensor, some_interaction_info = get_interaction_error(train_dict, tt_samples, zz2, rhott_samples,
                                                                  batch_size, disc_or_gen)

    # Finish computing the total loss
    if disc_or_gen == DISC_STRING:
        # Integral of phi_0 * rho_0
        disc_00_loss = get_disc_00_loss(discriminator, zero, batch_size, dim, device, Initial_set)
        # L2 Hamiltonian residual
        disc_ham_loss2 = torch.norm(hjb_error_tensor + lam_G*G_total_tensor, dim=1).mean(dim=0)
        # disc_ham_loss2 = sqeuc(hjb_error_tensor + lam_G*G_total_tensor).mean(dim=0)
        # Total loss
        total_loss = (-1) * (lam_disc_00 * disc_00_loss + lam_ham * ham_loss) + lam_ham2_d * disc_ham_loss2
        # total_loss = (-1) * (disc_00_loss) + lam_ham2_d * disc_ham_loss2
    elif disc_or_gen == GEN_STRING:
        # Integral of (psi - phi) * rho_T (but modified as regularizer)
        # gen_TT_loss = get_gen_TT_loss(generator, zz, TT, lam_gen_TT, device)
        # Total loss
        G_total_loss = G_total_tensor.mean(dim=0)
        # assert(G_total_loss.size() == torch.Size([1]))
        total_loss = lam_ham * ham_loss + lam_G * G_total_loss  # + lam_gen_TT * gen_TT_loss
    else:
        raise ValueError(error_msg)

    # Backprop
    total_loss.backward()
    disc_gen_grad_clip(discriminator, generator, grad_norm_clip_value, disc_or_gen)
    disc_gen_optimizer_step(disc_optimizer, gen_optimizer, disc_or_gen)


    ### Some Info ###
    with torch.no_grad():
        ham_loss2 = torch.norm(hjb_error_tensor + lam_G * G_total_tensor, dim=1).mean(dim=0)
        # ham_loss2 = sqeuc(hjb_error_tensor + lam_G * G_total_tensor).mean(dim=0)
        some_info_training = {'total_loss': total_loss.item(), 'ham_loss2': ham_loss2.item()}
        some_info_training['ham_loss'] = ham_loss.item()
        some_info_training['G_obstacle_loss'] = some_interaction_info['G_obstacle_loss']
        some_info_training['G_congestion_loss'] = some_interaction_info['G_congestion_loss']
        some_info_training['G_quadrun_loss'] = some_interaction_info['G_quadrun_loss']
        some_info_training['phi_lap_xx'] = some_hjb_error_info['phi_lap_xx']

        if disc_or_gen == DISC_STRING:
            some_info_training['disc_00_loss'] = disc_00_loss.item()
        elif disc_or_gen == GEN_STRING:
            pass  # some_info_training['gen_TT_loss'] = gen_TT_loss.item()
        else:
            raise ValueError(error_msg)

        if epoch % print_rate == 0:  #TODO: Try to move this outside of this function, cause the print_rate thingy goes against good coding philosophy
            ham_loss_max = torch.norm(hjb_error_tensor + lam_G * G_total_tensor, dim=1).max()  # TODO: Maybe put this in with the above
            some_info_training['ham_loss_max'] = ham_loss_max.item()

        # print('some_info_training:', some_info_training)

    return some_info_training

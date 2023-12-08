import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from os.path import join
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ===================================
# Utility functions and miscellaneous
# ===================================

DISC_STRING = 'disc'
GEN_STRING = 'gen'

sqeuc = lambda x: torch.sum(x * x, dim=1, keepdim=True)

# Some activation functions
act_funcs = {}
act_funcs['swish'] = lambda x: torch.sigmoid(x) * x
act_funcs['tanh'] = lambda x: torch.tanh(x)
act_funcs['lecun_tanh'] = lambda x: 1.7159 * torch.tanh((2 / 3) * x)
act_funcs['leaky_relu'] = lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.2)
act_funcs['relu'] = lambda x: torch.relu(x)
act_funcs['identity'] = lambda x: x
act_funcs['softplus'] = lambda x: torch.nn.functional.softplus(x)
act_funcs['leaky_softplus'] = lambda x: 0.2 * x + (1 - 0.2) * (torch.nn.functional.softplus(x) - torch.log(torch.tensor(2.0)))
act_funcs['leaky_softplus2'] = lambda x: 0.1 * x + (1 - 0.1) * (torch.nn.functional.softplus(x) - torch.log(torch.tensor(2.0)))
act_funcs['lin_tanh'] = lambda x: x + torch.tanh(x)
act_funcs['lin_lecun_tanh'] = lambda x: x + act_funcs['lecun_tanh'](x)
act_funcs['relu_tanh'] = lambda x: torch.relu(x)*0.1 + torch.tanh(x)*0.9
act_funcs['softplus2'] = lambda x: torch.nn.functional.softplus(x, beta=5)
act_funcs['softplus3'] = lambda x: torch.nn.functional.softplus(x, beta=10)
act_funcs['softplus_zero'] = lambda x: torch.nn.functional.softplus(x) - torch.log(torch.tensor(2.0))
act_funcs['softplus4'] = lambda x: torch.nn.functional.softplus(x, beta=3) - (1/3)*torch.log(torch.tensor(2.0))
act_funcs['softplus5'] = lambda x: torch.nn.functional.softplus(x, beta=3)
act_funcs['log_sigmoid'] = lambda x: torch.nn.functional.logsigmoid(x)

# ================================
# Uniform [0,1] sampler (for time)
# ================================

def uniform_time_sampler(batch_size, device):
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor(torch.Size([batch_size, 1])).uniform_()
    else:
        return torch.rand(size=(batch_size, 1)).to(device)


# =================
# Compute Laplacian
# =================

def get_laplacian2(grad, xx, batch_size, dim, grad_outputs_vec, device=torch.device('cpu')):
    hess_stripes = torch.autograd.grad(outputs=grad, inputs=xx,
                               grad_outputs=grad_outputs_vec,
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

    pre_laplacian = torch.stack([hess_stripes[i * batch_size: (i+1) * batch_size, i] for i in range(0, dim)], dim=1)
    laplacian = torch.sum(pre_laplacian, dim=1)
    laplacian_sum_repeat = laplacian.repeat(repeats=(1, dim))

    return laplacian_sum_repeat.T


# ==================================
#              Printing
# ==================================

# ====================
# Plotting the obstacle
# ====================

def _G_func_obstacle_np(x, y):
    """
    The final-time cost function.
    Returns a torch.tensor

    ONLY FOR 2D PLOTTING
    """
    # dim = x.shape[1]

    ## Symmetric obstacle
    # center = np.array([0, 0], dtype=np.float)
    # mat = np.array([[1, 0.8], [0.8, 1]], dtype=np.float)
    # vec = np.array([x, y], dtype=np.float) - center
    # quad = np.dot(vec, np.dot(mat, vec))
    # out = (-1) * quad + 0.1

    ## Non-symmetric obstacle
    # center = np.array([0, 1.5], dtype=np.float)
    # mat = np.array([[10, 0], [0, 0]], dtype=np.float)
    # bb = np.array([0, 5], dtype=np.float)
    # vec = np.array([x, y], dtype=np.float) - center
    # quad = np.dot(vec, np.dot(mat, vec))
    # lin = np.dot(vec, bb)
    # out = (-1) * (quad + lin) - 1

    ## Two obstacles
    # # Left obstacle
    # center1 = np.array([-2, 2], dtype=np.float)
    # vec1 = np.array([x, y], dtype=np.float) - center1
    # mat1 = np.array([[20, 0], [0, 0]], dtype=np.float)
    # bb1 = np.array([0, 5], dtype=np.float)
    # quad1 = np.dot(vec1, np.dot(mat1, vec1))
    # lin1 = np.dot(vec1, bb1)
    # out1 = np.clip((-1) * (quad1 + lin1) - 1, a_min=-0.1, a_max=None)
    #
    # # Right obstacle
    # center2 = np.array([2, -2], dtype=np.float)
    # vec2 = np.array([x, y], dtype=np.float) - center2
    # mat2 = np.array([[-20, 0], [0, 0]], dtype=np.float)
    # bb2 = np.array([0, 5], dtype=np.float)
    # quad2 = np.dot(vec2, np.dot(mat2, vec2))
    # lin2 = np.dot(vec2, bb2)
    # out2 = np.clip((quad2 + lin2) - 1, a_min=-0.1, a_max=None)
    #
    # out = out1 + out2

    ## Two diagonal obstacles
    # Rotation matrix
    theta = (np.pi / 5)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float)

    # Bottom/Left obstacle
    center1 = np.array([-2, 0.5], dtype=np.float)
    vec1 = np.array([x, y], dtype=np.float) - center1
    vec1 = np.dot(vec1, rot_mat)
    mat1 = np.array([[5, 0], [0, 0]], dtype=np.float)
    bb1 = np.array([0, 2], dtype=np.float)
    quad1 = np.dot(vec1, np.dot(mat1, vec1))
    lin1 = np.dot(vec1, bb1)
    out1 = np.clip((-1) * (quad1 + lin1 + 1), a_min=-0.1, a_max=None)

    # Top/Right obstacle
    center2 = np.array([2, -0.5], dtype=np.float)
    vec2 = np.array([x, y], dtype=np.float) - center2
    vec2 = np.dot(vec2, rot_mat)
    mat2 = np.array([[5, 0], [0, 0]], dtype=np.float)
    bb2 = np.array([0, -2], dtype=np.float)
    quad2 = np.dot(vec2, np.dot(mat2, vec2))
    lin2 = np.dot(vec2, bb2)
    out2 = np.clip((-1) * (quad2 + lin2 + 1), a_min=-0.1, a_max=None)

    out = out1 + out2

    ## Bottleneck
    # center = np.array([0, 0], dtype=np.float)
    # mat = np.array([[5, 0], [0, -1]], dtype=np.float)
    # vec = np.array([x, y], dtype=np.float) - center
    # quad = np.dot(vec, np.dot(mat, vec))
    # out = (-1) * quad - 0.1

    ## Bottleneck obstacle 2
    # # Bottom obstacle
    # center1 = np.array([0, 0.3], dtype=np.float)
    # vec1 = np.array([x, y], dtype=np.float) - center1
    # mat1 = np.array([[2, 0], [0, 0]], dtype=np.float)
    # bb1 = np.array([0, 2], dtype=np.float)
    # quad1 = np.dot(vec1, np.dot(mat1, vec1))
    # lin1 = np.dot(vec1, bb1)
    # out1 = np.clip((-1) * (quad1 + lin1 + 1), a_min=-0.1, a_max=None)
    #
    # # Top obstacle
    # center2 = np.array([0, -0.3], dtype=np.float)
    # vec2 = np.array([x, y], dtype=np.float) - center2
    # mat2 = np.array([[2, 0], [0, 0]], dtype=np.float)
    # bb2 = np.array([0, -2], dtype=np.float)
    # quad2 = np.dot(vec2, np.dot(mat2, vec2))
    # lin2 = np.dot(vec2, bb2)
    # out2 = np.clip((-1) * (quad2 + lin2 + 1), a_min=-0.1, a_max=None)
    #
    # out = out1 + out2

    return out



def make_obstacle_contour_plot(args):
    plot_window = args['plot_window']

    x = np.linspace(-plot_window, plot_window, 201)
    y = np.linspace(-plot_window, plot_window, 201)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    if args['lam_obstacle'] > 0:
        vec_func1 = np.vectorize(_G_func_obstacle_np)
        Z += args['lam_obstacle'] * vec_func1(X, Y)

    return X, Y, Z

# ================
# Making the plots
# ================

def make_plots(generator, discriminator, plot_dict, the_logger):
    """
    Prints trajectories for 8 timesteps. Blue points are the start (time=0), red points are the end (time=T).
    """
    epoch = plot_dict['epoch']
    print_rate = plot_dict['print_rate']
    num_plot_samples = plot_dict['num_plot_samples']
    zz_saved = plot_dict['zz_saved']  # plot_dict['zz_sampler']()
    zzz_saved = plot_dict['zzz_saved']
    TT = plot_dict['TT']
    device = plot_dict['device']
    obstacle_plot = plot_dict['obstacle_plot']
    plot_window = plot_dict['plot_window']

    with torch.no_grad():
        discriminator.eval()
        generator.eval()

        fig, ax = plt.subplots(1)
        ax.set(aspect='equal')

        # Plot contours of obstacle
        X = obstacle_plot[0]
        Y = obstacle_plot[1]
        Z = obstacle_plot[2]
        # 可恢复 obstacle_contours = plt.contour(X, Y, Z, levels=[0], colors='C3', zorder=1)
        # obstacle_contours = plt.contour(X, Y, Z, zorder=1)
        # plt.clabel(obstacle_contours)

        # Plot contours of phi
        disc_time_contour = 0.5
        X1 = X.reshape(-1, 1)
        Y1 = Y.reshape(-1, 1)
        dim = zz_saved.size(1)
        XY1 = np.concatenate((X1, Y1, 1*np.ones(shape=(X1.shape[0], dim-2))), axis=1)
        XY1 = torch.tensor(XY1, dtype=torch.float)
        Z_disc = discriminator(torch.tensor([disc_time_contour], dtype=torch.float).expand(XY1.size(0), 1).to(device),
                               XY1.to(device))
        Z_disc = Z_disc.cpu().numpy().reshape(Z.shape)
        # 可恢复 disc_contours = plt.contour(X, Y, Z_disc, alpha=0.5, zorder=0)
        # 可恢复 plt.clabel(disc_contours)
        # plt.colorbar()

        # Plot the agents
        timesteps = 8
        color = iter(cm.rainbow(np.linspace(0, 1, timesteps + 2)))
        c = next(color)
        for idx in range(0, timesteps + 1):
            c = next(color)
            the_timepoint = torch.tensor([(TT[0].item() / timesteps) * (idx)], dtype=torch.float).to(device)
            plot_sample = generator(the_timepoint[0].expand(num_plot_samples, 1), zz_saved, zzz_saved).cpu().detach().numpy()
            # plot_sample += np.random.normal(loc=[0,0], scale=0.1, size=plot_sample.shape)
            # alpha = 1 if (idx == 0 or idx == timesteps) else 0.3
            alpha = 0.5
            ax.scatter(plot_sample[:, 0], plot_sample[:, 1], color=c, alpha=alpha, zorder=2)

        if dim > 2:
            avg_terminal_point = plot_sample.mean(axis=0)
            print('avg term point other dims:', avg_terminal_point[2:])

            dist_zz_saved_and_term = np.linalg.norm(zz_saved[:, 2:].cpu().numpy() - plot_sample[:, 2:], axis=1)
            avg_dist_zz_saved_and_term = dist_zz_saved_and_term.mean(axis=0)
            print('avg normdist of term from start other dims:', avg_dist_zz_saved_and_term)

        plt.title(f'epoch: {epoch}')
        ax.set_xlim([-plot_window, plot_window])
        ax.set_ylim([-plot_window, plot_window])

        if plot_dict['show_plots']:
            plt.show()

        the_logger.save_plot(epoch, fig)

        plt.close(fig)

        discriminator.train()
        generator.train()


# ===========================
# Lambda HJB error scheduling
# ===========================

class MultiStepScheduler(object):

    def __init__(self, initial_value, milestones, gamma=0.1):
        self.current_value = initial_value
        self.milestones = list(milestones)
        self.gamma = gamma
        self.count = 0

    def step(self):
        self.count += 1
        if len(self.milestones) > 0 and self.count == self.milestones[0]:
            self.current_value *= self.gamma
            self.milestones.pop(0)
        return self.current_value

    def get(self):
        return self.current_value

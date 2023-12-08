import numpy as np
import torch
import math
from utils.utils import sqeuc, DISC_STRING, GEN_STRING

# ============
# Hamiltonians
# ============

# Hamiltonian function
ham_funcs = {}
ham_funcs['hom1'] = lambda tt, xx, grad: torch.norm(grad, dim=1, keepdim=True)
ham_funcs['hom2'] = lambda tt, xx, grad: 0.5 * sqeuc(grad)

# ==============
# Psi function
# ==============

def psi_func_torch(x, device, scale=1):
    """
    The final-time cost function.
    Returns a torch.tensor
    """
    xx = x[:, 0:2]
    dim = xx.size(1)
    assert(dim == 2), f"Oops, dim is not 2. Instead dim={dim}"
    ## For congestion and non-symmetric obstacle
    # center = torch.tensor([[2, 0]], dtype=torch.float).to(device)

    ## For two diagonal obstacles 可恢复的
    #center = torch.tensor([[2, 2]], dtype=torch.float).to(device)

    ## For bottleneck
    # center = torch.tensor([[2, 0]], dtype=torch.float).to(device)

    # L2
    #out = scale * (sqeuc(xx - center))

    # L1 可恢复的
    #out = scale * torch.norm(xx - center, dim=1, keepdim=True)

    ##我的
    target_samples = torch.tensor([[0,0],[1.5,1.5]], dtype=torch.float).to(device)  # 终点分布
    out = torch.zeros(xx.size(0),1).to(device)
    for i in range(xx.size(0)):
        if xx[i,:] in target_samples:
            out[i,0] = 0
        else:
            out[i,0] = -1000

    return out


# =============
# Initial rho_0
# =============

def sample_rho0(Initial_set, num_samples, dim, var_scale=1/10):
    """
    The initial distribution rho_0 of the agents.
    Returns a numpy array.
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ## For symmetric obstacle
    # mu = torch.tensor([[-1] * dim], dtype=torch.float)  # for barrier problem
    # out = math.sqrt(var_scale) * torch.randn(size=(num_samples, dim)) + mu

    ## For congestion and non-symmetric obstacle
    # mu = torch.tensor([[-2, 0] + [0] * (dim - 2)], dtype=torch.float)
    # out = math.sqrt(var_scale) * torch.randn(size=(num_samples, dim)) + mu

    ## For two obstacles
    # mu = torch.tensor([[-5, 0] + [-5] * (dim - 2)], dtype=torch.float)
    # out = math.sqrt(var_scale) * torch.randn(size=(num_samples, dim)) + mu

    ## For two diagonal obstacles 需要恢复的
    #mu = torch.tensor([[-2, -2] + [-2] * (dim - 2)], dtype=torch.float)
    #out = math.sqrt(var_scale) * torch.randn(size=(num_samples, dim)) + mu

    ## For bottleneck obstacle
    # mu = torch.tensor([[-2, 0] + [-2] * (dim - 2)], dtype=torch.float)
    # out = math.sqrt(var_scale) * torch.randn(size=(num_samples, dim)) + mu

    ## For (plain) congestion
    # mu = torch.tensor([[-1, 0] + [-1] * (dim - 2)], dtype=torch.float)
    # out = math.sqrt(var_scale) * torch.randn(size=(num_samples, dim)) + mu

    # For Minist
    # prob = image.view(-1, image.shape[0] * image.shape[1]).float()
    # a = torch.multinomial(prob, num_samples=num_samples, replacement=True).float()
    # out = list()
    # for i in range(a.shape[1]):
    #     p = ((a[0, i] / (image.shape[0])).floor() - image.shape[0]*0.5)/14*3
    #     q = (a[0, i] % image.shape[1] - image.shape[1]*0.5)/14*3
    #     out.append([p, q])
    # out = torch.tensor(out)
    # For Minist Image as a point
    out = torch.empty(num_samples, Initial_set.shape[1])
    index = 0
    for i in range(num_samples):
        rand_index = torch.randint(0, Initial_set.shape[0], (1, 1))
        out[index, :] = Initial_set[rand_index]
        index += 1

    return out


def sample_rho0_fast_cuda(num_samples, dim, var_scale=1/10):
    """
    Creates tensor directly on GPU. Helps a tiny bit.
    """
    raise ValueError("DO NOT USE THIS")
    # return torch.cuda.FloatTensor(torch.Size([num_samples, dim])).normal_(mean=-1, std=math.sqrt(var_scale))


# =================================
# The G function / interaction term
# =================================

def G_func_obstacle_torch(x, device, scale=1):
    """
    Calculate interaction term. Calculates G(x), where G is a quadratic-type barrier.
    Returns a torch.tensor.
    This is a "cylindrical barrier," with the barrier in the first two dimensions.
    """
    batch_size = x.size(0)
    xx = x[:, 0:2]
    dim = xx.size(1)
    assert(dim == 2), f"Require dim=2 but, got dim={dim} (BAD)"

    ## Symmetric obstacle
    # mu = torch.tensor([0] * dim, dtype=torch.float).to(device)
    # covar_mat = torch.eye(dim, dtype=torch.float)
    # covar_mat[0:2, 0:2] = torch.tensor(np.array([[1, 0.8], [0.8, 1]]),
    #                                    dtype=torch.float)
    # covar_mat = covar_mat.expand(batch_size, dim, dim).to(device)
    #
    # xxmu = (xx - mu).unsqueeze(1).bmm(covar_mat)
    #
    # out = (-1) * torch.bmm(xxmu, (xx - mu).unsqueeze(2)) + 0.1
    # out = scale * out.view(-1, 1)
    # out = torch.relu(out)

    ## Non-symmetric obstacle
    # center = torch.tensor([0, 1.5], dtype=torch.float).to(device)
    # covar_mat = torch.eye(dim, dtype=torch.float)
    # covar_mat[0:2, 0:2] = torch.tensor(np.array([[10, 0], [0, 0]]),
    #                                    dtype=torch.float)
    # covar_mat = covar_mat.expand(batch_size, dim, dim).to(device)
    # bb_vec = torch.tensor([0, 5], dtype=torch.float).expand(xx.size()).to(device)
    #
    # xxcov = (xx - center).unsqueeze(1).bmm(covar_mat)
    #
    # quad = torch.bmm(xxcov, (xx - center).unsqueeze(2)).view(-1, 1)
    # lin = torch.sum((xx - center) * bb_vec, dim=1, keepdim=True)
    # out = (-1)*(quad + lin) - 1
    # out = scale * out.view(-1, 1)
    # out = torch.relu(out)

    ## Two obstacles
    # # Left obstacle
    # center1 = torch.tensor([-2, 2] + [0] * (dim - 2), dtype=torch.float).to(device)
    # covar_mat1 = torch.eye(dim, dtype=torch.float)
    # covar_mat1[0:2, 0:2] = torch.tensor(np.array([[20, 0], [0, 0]]), dtype=torch.float)
    # covar_mat1 = covar_mat1.expand(batch_size, dim, dim).to(device)
    # bb_vec1 = torch.tensor([0, 5], dtype=torch.float).expand(xx.size()).to(device)
    # xxcov1 = (xx - center1).unsqueeze(1).bmm(covar_mat1)
    # quad1 = torch.bmm(xxcov1, (xx - center1).unsqueeze(2)).view(-1, 1)
    # lin1 = torch.sum((xx - center1) * bb_vec1, dim=1, keepdim=True)
    # out1 = (-1) * (quad1 + lin1) - 1
    # out1 = scale * out1.view(-1, 1)
    # out1 = torch.clamp_min(out1, min=0)
    #
    # # Right obstacle
    # center2 = torch.tensor([2, -2] + [0] * (dim - 2), dtype=torch.float).to(device)
    # covar_mat2 = torch.eye(dim, dtype=torch.float)
    # covar_mat2[0:2, 0:2] = torch.tensor(np.array([[-20, 0], [0, 0]]), dtype=torch.float)
    # covar_mat2 = covar_mat2.expand(batch_size, dim, dim).to(device)
    # bb_vec2 = torch.tensor([0, 5], dtype=torch.float).expand(xx.size()).to(device)
    # xxcov2 = (xx - center2).unsqueeze(1).bmm(covar_mat2)
    # quad2 = torch.bmm(xxcov2, (xx - center2).unsqueeze(2)).view(-1, 1)
    # lin2 = torch.sum((xx - center2) * bb_vec2, dim=1, keepdim=True)
    # out2 = (quad2 + lin2) - 1
    # out2 = scale * out2.view(-1, 1)
    # out2 = torch.clamp_min(out2, min=0)
    #
    # out = out1 + out2

    # Two diagonal obstacles
    # Rotation matrix
    theta = torch.tensor(np.pi / 5)
    rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta), torch.cos(theta)]]).expand(batch_size, dim, dim).to(device)

    # Bottom/Left obstacle  # TODO: Clean it up
    center1 = torch.tensor([-2, 0.5], dtype=torch.float).to(device)
    xxcent1 = xx - center1
    xxcent1 = xxcent1.unsqueeze(1).bmm(rot_mat).squeeze(1)
    covar_mat1 = torch.eye(dim, dtype=torch.float)
    covar_mat1[0:2, 0:2] = torch.tensor(np.array([[5, 0], [0, 0]]), dtype=torch.float)
    covar_mat1 = covar_mat1.expand(batch_size, dim, dim).to(device)
    bb_vec1 = torch.tensor([0, 2], dtype=torch.float).expand(xx.size()).to(device)
    xxcov1 = xxcent1.unsqueeze(1).bmm(covar_mat1)
    quad1 = torch.bmm(xxcov1, xxcent1.unsqueeze(2)).view(-1, 1)
    lin1 = torch.sum(xxcent1 * bb_vec1, dim=1, keepdim=True)
    out1 = (-1) * ((quad1 + lin1) + 1)
    out1 = scale * out1.view(-1, 1)
    out1 = torch.clamp_min(out1, min=0)

    # Top/Right obstacle
    center2 = torch.tensor([2, -0.5], dtype=torch.float).to(device)
    xxcent2 = xx - center2
    xxcent2 = xxcent2.unsqueeze(1).bmm(rot_mat).squeeze(1)
    covar_mat2 = torch.eye(dim, dtype=torch.float)
    covar_mat2[0:2, 0:2] = torch.tensor(np.array([[5, 0], [0, 0]]), dtype=torch.float)
    covar_mat2 = covar_mat2.expand(batch_size, dim, dim).to(device)
    bb_vec2 = torch.tensor([0, -2], dtype=torch.float).expand(xx.size()).to(device)
    xxcov2 = xxcent2.unsqueeze(1).bmm(covar_mat2)
    quad2 = torch.bmm(xxcov2, xxcent2.unsqueeze(2)).view(-1, 1)
    lin2 = torch.sum(xxcent2 * bb_vec2, dim=1, keepdim=True)
    out2 = (-1) * ((quad2 + lin2) + 1)
    out2 = scale * out2.view(-1, 1)
    out2 = torch.clamp_min(out2, min=0)

    out = out1 + out2

    ## Bottleneck obstacle
    # center = torch.tensor([0, 0] + [0] * (dim - 2), dtype=torch.float).to(device)
    # covar_mat = torch.eye(dim, dtype=torch.float)
    # covar_mat[0:2, 0:2] = torch.tensor(np.array([[5, 0], [0, -1]]),
    #                                    dtype=torch.float)
    # covar_mat = covar_mat.expand(batch_size, dim, dim).to(device)
    #
    # xxmu = (xx - center).unsqueeze(1).bmm(covar_mat)
    #
    # out = (-1) * torch.bmm(xxmu, (xx - center).unsqueeze(2)) - 0.1
    # out = scale * out.view(-1, 1)
    # out = torch.relu(out)

    ## Bottleneck obstacle 2
    # # Bottom obstacle  # TODO: Clean it up
    # center1 = torch.tensor([0, 0.3], dtype=torch.float).to(device)
    # xxcent1 = xx - center1
    # covar_mat1 = torch.eye(dim, dtype=torch.float)
    # covar_mat1[0:2, 0:2] = torch.tensor(np.array([[2, 0], [0, 0]]), dtype=torch.float)
    # covar_mat1 = covar_mat1.expand(batch_size, dim, dim).to(device)
    # bb_vec1 = torch.tensor([0, 2], dtype=torch.float).expand(xx.size()).to(device)
    # xxcov1 = xxcent1.unsqueeze(1).bmm(covar_mat1)
    # quad1 = torch.bmm(xxcov1, xxcent1.unsqueeze(2)).view(-1, 1)
    # lin1 = torch.sum(xxcent1 * bb_vec1, dim=1, keepdim=True)
    # out1 = (-1) * ((quad1 + lin1) + 1)
    # out1 = scale * out1.view(-1, 1)
    # out1 = torch.clamp_min(out1, min=0)
    #
    # # Top obstacle
    # center2 = torch.tensor([0, -0.3], dtype=torch.float).to(device)
    # xxcent2 = xx - center2
    # covar_mat2 = torch.eye(dim, dtype=torch.float)
    # covar_mat2[0:2, 0:2] = torch.tensor(np.array([[2, 0], [0, 0]]), dtype=torch.float)
    # covar_mat2 = covar_mat2.expand(batch_size, dim, dim).to(device)
    # bb_vec2 = torch.tensor([0, -2], dtype=torch.float).expand(xx.size()).to(device)
    # xxcov2 = xxcent2.unsqueeze(1).bmm(covar_mat2)
    # quad2 = torch.bmm(xxcov2, xxcent2.unsqueeze(2)).view(-1, 1)
    # lin2 = torch.sum(xxcent2 * bb_vec2, dim=1, keepdim=True)
    # out2 = (-1) * ((quad2 + lin2) + 1)
    # out2 = scale * out2.view(-1, 1)
    # out2 = torch.clamp_min(out2, min=0)
    #
    # out = out1 + out2
    out = torch.tensor([[0]]).to(device)
    return out


def G_congestion_torch4(generator, tt_samples, zz2, rhott_samples, disc_or_gen, first_d_dim=2, scale=1):
    """
    Interaction term, for congestion.
    Returns a torch.tensor.
    """
    if disc_or_gen == DISC_STRING:
        rhott_samples2 = generator(tt_samples, zz2).detach()
    elif disc_or_gen == GEN_STRING:
        rhott_samples2 = generator(tt_samples, zz2)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be \'disc\' or \'gen\' but got: {disc_or_gen}')
    rhott_samples_first_d = rhott_samples[:, :first_d_dim]
    rhott_samples2_first_d = rhott_samples2[:, :first_d_dim]

    distances = sqeuc(rhott_samples_first_d - rhott_samples2_first_d)
    out = 1 / (distances + 1)

    return out

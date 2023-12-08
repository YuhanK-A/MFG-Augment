import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# ==============================================
#            Make folder for training
# ==============================================

def _get_time_string():
    """
    Just get the current time in a string
    """
    ## New format, but messes with legacy
    out = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]
    out = out[:10] + '__' + out[11:]  # separate year-month-day from hour-minute-seconds

    # Old format
    # out = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]

    return out


folder_path = os.path.join('Experiments', 'Run_' + _get_time_string())
models_path = os.path.join(folder_path, 'models')

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    os.makedirs(models_path)

# =============================================
#           Define the neural network
# =============================================

class Hamiltonian_NN(torch.nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.lin1 = torch.nn.Linear(dim, 10)
        self.linlast = torch.nn.Linear(10, 1)

    def forward(self, inp):
        out = self.lin1(inp)
        out = F.relu(out)
        out = self.linlast(out)

        return out


# =========================================
#           Setup Hyperparameters
# =========================================

dim = 2
batch_size = 50
window_size = 5
max_epochs = 1000
learning_rate = 1e-2
momentum = 0.9
print_rate = 100

ham_nn = Hamiltonian_NN(dim=2)

optimizer = torch.optim.SGD(ham_nn.parameters(), lr=learning_rate, momentum=0.9)

criterion = torch.nn.MSELoss()

# =====================================
#                 Train
# =====================================

for idx in range(0, max_epochs):

    data = -window_size + (2 * window_size) * torch.rand(size=(batch_size, dim))

    generated = ham_nn(data)

    labels = torch.norm(data, p=1, dim=1, keepdim=True)

    loss = criterion(generated, labels)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if idx % print_rate == 0:
        print('loss:', loss.detach().numpy())

ham_nn.eval()

# ========================================================
#           Save the Hamiltonian neural networks
# ========================================================

save_dict = {'state_dict': ham_nn.state_dict()}

torch.save(save_dict, os.path.join(models_path, 'ham_nn.pth.tar'))

# ====================================================================================
#           Loading the Hamiltonian neural network we just trained as a test
# ====================================================================================

ham_nn2 = Hamiltonian_NN(dim=dim)

ham_nn_load = torch.load(os.path.join(models_path, 'ham_nn.pth.tar'), map_location=lambda storage, location: storage)

ham_nn2.load_state_dict(ham_nn_load['state_dict'])

ham_nn2.eval()

# ======================================
#               Make plots
# ======================================

arange = np.arange(start=-10, stop=10 + 0.01, step=0.01)
# print('xx:', xx)

XX, YY = np.meshgrid(arange, arange)
XX1, YY1 = XX.reshape(-1, 1), YY.reshape(-1, 1)
XX_YY1 = np.concatenate((XX1, YY1), axis=1)

Z_ham = ham_nn2(torch.tensor(XX_YY1, dtype=torch.float)).detach().numpy()
Z_ham = Z_ham.reshape(XX.shape)

Z_true = np.linalg.norm(XX_YY1, ord=1, axis=1)
Z_true = Z_true.reshape(XX.shape)

plt.contour(XX, YY, Z_ham)
# plt.contour(XX, YY, Z_true)
plt.show()
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


L = 3
D = int(1e4)
N = int(1e4)

N_test= int(256)
lr = 1e-1
EPOCHS = int(1e6)
losses_array = np.zeros(EPOCHS)
losses_validation_array = np.zeros(EPOCHS//100)


u_W_star = (torch.randn(D, requires_grad=False)).to(device)
u_V_star = (torch.randn(D, requires_grad=False)).to(device)
"""with torch.no_grad():
    W_star = torch.outer(u_W_star, u_W_star)
    V_star = torch.outer(u_V_star, u_V_star)"""
gamma = 0.49

activation_function_str = 'softmax'  # Change to 'softplus' or 'lin' as needed


def activation_function(tensor, activation_func_str='softmax'):
        if activation_func_str == 'softmax':
            return F.softmax(tensor, dim=-1)
        elif activation_func_str == 'softplus':
            eps = 2e-30
            sigma = F.softplus(tensor)
            denominator = torch.sum(sigma, dim=-1, keepdim=True) + eps
            return sigma / denominator
        elif activation_func_str == 'lin':
            return 1 + tensor
        else:
            raise ValueError(f"Unsupported activation function: {activation_func_str}")

def sample_tilted_X_fast(batch_size, L, d, u_W_star, gamma, eps1_idx=0, eps2_idx=1):
    with torch.no_grad():
        X = torch.randn(batch_size, L, d, device=device)
        norm_u_W_star = torch.linalg.vector_norm(u_W_star)
        rho = gamma * norm_u_W_star**2 / d
        e = u_W_star / norm_u_W_star
      

        Sigma = torch.tensor([
            [1, rho],
            [rho, 1]
        ], device=device) / (1 - rho**2)

        dist = torch.distributions.MultivariateNormal(torch.zeros(2, device=device), covariance_matrix=Sigma)
        ps = dist.sample((batch_size,))
        s1 = ps[:, 0]
        s2 = ps[:, 1]

        z1 = torch.randn(batch_size, d, device=device)
        z2 = torch.randn(batch_size, d, device=device)
        
        z1 -= (z1 @ e).unsqueeze(1) * e.unsqueeze(0)
        z2 -= (z2 @ e).unsqueeze(1) * e.unsqueeze(0)

        X[:, eps1_idx, :] = s1[:, None] * e.reshape(1, d) + z1
        X[:, eps2_idx, :] = s2[:, None] * e.reshape(1, d) + z2

    return X

def instance_spiked_fast(gamma, u_W_star, u_V_star, N, L, D):
    with torch.no_grad():
        eps1_idx = 0
        eps2_idx = 1
        X = sample_tilted_X_fast(N, L, D, u_W_star, gamma, eps1_idx, eps2_idx)
        X_eps1 = X[:, eps1_idx, :]
        X_eps2 = X[:, eps2_idx, :]
        
        
        dot1 = torch.einsum('nd,d->n', X_eps1, u_V_star)
        dot2 = torch.einsum('nd,d->n', X_eps2, u_V_star)
        y = (dot1 * dot2) / D
        
    return X, y

class EmpiricalRisk_Model(nn.Module):
    def __init__(self, D, activation_function_str='softmax', u_W1=None, u_W2=None, u_V=None):
        super(EmpiricalRisk_Model, self).__init__()
        self.D = D
        
        init_scale = 1
        self.u_W1 = nn.Parameter(torch.randn(D)*init_scale if u_W1 is None else u_W1.clone())
        self.u_V = nn.Parameter(torch.randn(D)*init_scale if u_V is None else u_V.clone())
        self.u_W2 = nn.Parameter(torch.randn(D)*init_scale if u_W2 is None else u_W2.clone())
        self.activation_function_str = activation_function_str

    def get_Xu(self, X, u):
        return torch.einsum('nld,d->nl', X, u/(self.D**0.5))
    
    def get_XuXuT(self, Xu):
        return torch.einsum('nl,nm->nlm', Xu, Xu)

    def forward(self, x):

        Xu_W1 = self.get_Xu(x, self.u_W1)
        Xu_W2 = self.get_Xu(x, self.u_W2)
        Xu_V = self.get_Xu(x, self.u_V)
        A1 = self.get_XuXuT(Xu_W1)
        

        A2 = self.get_XuXuT(Xu_W2)
        
        B = self.get_XuXuT(Xu_V)
        

        act1 = activation_function(A1, self.activation_function_str)[:, 0, :]
        act2 = activation_function(A2, self.activation_function_str)[:, 0, :]


        res = torch.einsum('ni,nij,nj->n', act1, B, act2)

        return res

Model = EmpiricalRisk_Model(D, activation_function_str=activation_function_str).to(device)

with torch.no_grad():
    u_W1_W_star = torch.dot(Model.u_W1, u_W_star)/D
    u_W2_W_star = torch.dot(Model.u_W2, u_W_star)/D
    u_V_W_star = torch.dot(Model.u_V, u_W_star)/D

    u_W1_V_star = torch.dot(Model.u_W1, u_V_star)/D
    u_W2_V_star = torch.dot(Model.u_W2, u_V_star)/D
    u_V_V_star = torch.dot(Model.u_V, u_V_star)/D

    u_W1_W1 = torch.dot(Model.u_W1, Model.u_W1)/D
    u_W2_W2 = torch.dot(Model.u_W2, Model.u_W2)/D
    u_V_V = torch.dot(Model.u_V, Model.u_V)/D

    u_W1_V = torch.dot(Model.u_W1, Model.u_V)/D
    u_W2_V = torch.dot(Model.u_W2, Model.u_V)/D
    u_W1_W2 = torch.dot(Model.u_W1, Model.u_W2)/D

    print(f"u_W1_W_star: {u_W1_W_star.item()}, u_W2_W_star: {u_W2_W_star.item()}, u_V_W_star: {u_V_W_star.item()}", flush=True)
    print(f"u_W1_V_star: {u_W1_V_star.item()}, u_W2_V_star: {u_W2_V_star.item()}, u_V_V_star: {u_V_V_star.item()}", flush=True)
    print(f"u_W1_W1: {u_W1_W1.item()}, u_W2_W2: {u_W2_W2.item()}, u_V_V: {u_V_V.item()}", flush=True)
    print(f"u_W1_V: {u_W1_V.item()}, u_W2_V: {u_W2_V.item()}, u_W1_W2: {u_W1_W2.item()}", flush=True)

def Compute_Loss(model, X, y_true):
    y_pred = model(X)
    loss =   F.mse_loss(y_pred, y_true)
    return loss

X_val, y_val = instance_spiked_fast(gamma=gamma, u_W_star=u_W_star,u_V_star=u_V_star ,N=N_test, L=L, D=D)
X_val = X_val.to(device)
y_val = y_val.to(device)

X_test, y_test = instance_spiked_fast(gamma=gamma, u_W_star=u_W_star,u_V_star=u_V_star ,N=N_test, L=L, D=D)
X_test = X_test.to(device)
y_test = y_test.to(device)

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        loss = Compute_Loss(model, X, y)
    return loss.item()

def train_model(model, epochs):
    for epoch in range(epochs):
        model.train()
        X, y = instance_spiked_fast(gamma=gamma, u_W_star=u_W_star, u_V_star=u_V_star, N=N, L=L, D=D)
        X = X.to(device)
        y = y.to(device)
        
        loss = Compute_Loss(model, X, y)
        losses_array[epoch] = loss.item()
        loss.backward()
        with torch.no_grad():
            model.u_W1 -= lr * model.u_W1.grad
            model.u_W1.grad.zero_()

            model.u_W2 -= lr * model.u_W2.grad
            model.u_W2.grad.zero_()

            model.u_V -= lr * model.u_V.grad
            model.u_V.grad.zero_()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            with torch.no_grad():
                eval_loss = evaluate_model(model, X_val, y_val)
                losses_validation_array[epoch//100] = eval_loss
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {eval_loss}", flush=True)
    
    torch.save(model.state_dict(), 'model_state_dict_EPOCHS_1e6_d_1e4_N_1e4_lr_1minus1.pt')

    

def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        loss = Compute_Loss(model, X_test, y_test)
        print(f"Test Loss: {loss.item()}", flush=True)

if __name__ == "__main__":
    train_model(Model, EPOCHS)
    test_model(Model, X_test, y_test)

    np.savetxt('training_loss_EPOCHS_1e6_d_1e4_seed_0_batch_1e4_lr_1minus1.txt', losses_array)
    np.savetxt('validation_loss_EPOCHS_1e6_d_1e4_seed_0_batch_1e4_lr_1minus1.txt', losses_validation_array)
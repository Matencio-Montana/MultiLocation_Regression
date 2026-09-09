import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L= 3
a = 1.0 
L_below_flat = torch.nn.Parameter(torch.tensor([0.0,0.0,a, 0.0,0.0,0.0,a, 0.0,0.0,0.0,0.0,a], device=device, requires_grad=True))
nMc= int(1e6)
Zs = torch.randn(nMc,L*5, device=device)
activation_func_str='softmax'
gamma = 0.49


def build_L_block_unit_variance(theta, eps=2e-30):
    
    
    r3 = theta[0:3]
    r4 = theta[3:7]
    r5 = theta[7:12]

   

    L_block = torch.zeros((5, 5), dtype=theta.dtype, device=theta.device)

   
    L_block[0, 0] = 1.0
    L_block[1, 1] = 1.0

    
    L_block[2, 0] = r3[0]
    L_block[2, 1] = r3[1]
    L_block[2, 2] = r3[2] 

    L_block[3, 0] = r4[0]
    L_block[3, 1] = r4[1]
    L_block[3, 2] = r4[2]
    L_block[3, 3] = r4[3] 

    L_block[4, 0] = r5[0]
    L_block[4, 1] = r5[1]
    L_block[4, 2] = r5[2]
    L_block[4, 3] = r5[3]
    L_block[4, 4] = r5[3] 

    return L_block

def get_Xus(L_below_flat, Zs):
    Chol_block = build_L_block_unit_variance(L_below_flat)
    Cholesky = torch.block_diag(*[Chol_block]*L)
    Xus = torch.einsum('ab,nb->na', Cholesky, Zs).reshape(-1, L, 5)

    X_u_W_star = Xus[:,:, 0]
    X_u_V_star = Xus[:,:, 1]
    X_u_W_1 = Xus[:,:, 2]
    X_u_V = Xus[:,:, 3]
    X_u_W_2 = Xus[:,:, 4]

    return X_u_W_star, X_u_V_star, X_u_W_1, X_u_V, X_u_W_2


def get_chi_star(X_u_W_star):
    chi_star = torch.einsum('...p,...q->...pq', X_u_W_star, X_u_W_star)
    return chi_star

def get_g(chi_star, gamma, epsilon_star_1=0, epsilon_star_2=1):
    g = F.softmax(gamma * chi_star.view(-1, L*L), dim=-1).view(-1, L, L)
    g = g[:, epsilon_star_1, epsilon_star_2]
    return g

def get_y_true(X_u_V_star, epsilon_star_1=0, epsilon_star_2=1):
    return X_u_V_star[:, epsilon_star_1] * X_u_V_star[:, epsilon_star_2]

def activation_function(tensor, activation_func_str='softmax'):
    if activation_func_str == 'softmax':
        return F.softmax(tensor, dim=-1)
    elif activation_func_str == 'softplus':
        eps=2e-30
        sigma = F.softplus(tensor)
        denominator = torch.sum(sigma, dim=-1, keepdim=True) + eps
        return sigma/denominator
    elif activation_func_str == 'lin':
        return 1+tensor
    else:
        raise ValueError(f"Unsupported activation function: {activation_func_str}")

def pop_err(L_below_flat):
    X_u_W_star, X_u_V_star, X_u_W_1, X_u_V, X_u_W_2 = get_Xus(L_below_flat, Zs)
    y_true = get_y_true(X_u_V_star).to(device)
    g = get_g(get_chi_star(X_u_W_star), gamma=gamma).to(device)


    A1 = torch.einsum('ni,nj->nij', X_u_W_1, X_u_W_1)
    A2 = torch.einsum('ni,nj->nij', X_u_W_2, X_u_W_2)
    B = torch.einsum('ni,nj->nij', X_u_V, X_u_V)

    act1 = activation_function(A1, activation_func_str=activation_func_str)[...,0,:]
    act2 = activation_function(A2, activation_func_str=activation_func_str)[...,0,:]
    l2= (y_true - torch.einsum('ni,nij,nj->n', act1, B, act2) )**2
    res = torch.mean(g * l2)/torch.mean(g) 
    return res

lbfgs_steps = int(1e3)
optimizer = torch.optim.LBFGS(
    [L_below_flat],
    lr=1.0,
    max_iter=20,
    history_size=100,
    line_search_fn='strong_wolfe'
)

for iteration in range(lbfgs_steps):
    def closure():
        optimizer.zero_grad()
        loss = pop_err(L_below_flat)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        err = pop_err(L_below_flat)
    print(f"Iteration {iteration}, Optimization error: {err.item()}", flush=True)

print(f"Optimization error: {err.item()}", flush=True)

L_block_opt = build_L_block_unit_variance(L_below_flat).cpu().detach().numpy()
Q_opt = L_block_opt @ L_block_opt.T
np.save('L_block_opt.npy', L_block_opt)
np.save('Q_opt.npy', Q_opt)
print(f"Optimized Q_opt:\n{Q_opt}", flush=True)


import torch
import torch.nn as nn

class DDPM(nn.Module):
    
    def __init__(self, n_steps, min_beta, max_beta, device):
        super().__init__()
        
        self.device = device
        self.n_steps = n_steps
        
        self.betas = torch.linspace(start=min_beta, end=max_beta, steps=n_steps, device=device)
        self.alphas = 1 - self.betas
        
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # pre-expand coefficients to [T, 1, 1, 1] for fixed-size image tensors.
        self.betas_4d = self.betas.view(n_steps, 1, 1, 1)
        self.alphas_4d = self.alphas.view(n_steps, 1, 1, 1)
        self.alpha_bars_4d = self.alpha_bars.view(n_steps, 1, 1, 1)
        self.one_tensor = torch.tensor(1.0, device=self.device)
    
    def sample_forward_step(self, x_t_minus_1, t, eps=None):
        # one step forward
        # x_t = sqrt(1 - beta) * x_t_minus_1 + sqrt(beta) * epsilon
        if eps is None:
            eps = torch.randn_like(x_t_minus_1, device=self.device)
        
        alpha_t = self.alphas_4d[t]
        beta_t = self.betas_4d[t]

        x_t = torch.sqrt(alpha_t) * x_t_minus_1 + torch.sqrt(beta_t) * eps
        
        return x_t
    
    def sample_forward(self, x_0, t, eps=None):
        # t steps forward, but can be calculated in one step
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        if eps is None:
            eps = torch.randn_like(x_0, device=self.device) # sample from normal distribution        
        
        alpha_bar_t = self.alpha_bars_4d[t]
            
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps
        
        return x_t

    def sample_backward_step(self, x_t, t, eps, simple_var = True):
        if eps is None:
            eps = torch.randn_like(x_t, device=self.device) # usually epsilon is something we need to predict
        
        alpha_t = self.alphas_4d[t]
        alpha_bar_t = self.alpha_bars_4d[t]
        alpha_bar_t_minus_1 = self.alpha_bars_4d[t - 1] if t > 0 else self.one_tensor
        beta_t = self.betas_4d[t]
        
        if t == 0:
            noise = 0
        else:
            variance_t = beta_t if simple_var else \
                         beta_t * (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t)
            z = torch.randn_like(x_t, device=self.device)
            noise = torch.sqrt(variance_t) * z
        
        mean = (x_t - (1 - alpha_t) * eps / torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_t)
        
        x_t_minus_1 = mean + noise
        
        return x_t_minus_1

    def sample_backward(self, x, eps_pred):
        
        for t in range(self.n_steps - 1, -1, -1):
            eps_t = eps_pred(x, t)
            x = self.sample_backward_step(x, t, eps_t)
        
        return x
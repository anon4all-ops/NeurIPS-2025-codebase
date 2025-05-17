import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
from autoattack import AutoAttack
import torchattacks
import torch.distributions as dist

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack


def sample_delta(x, epsilon):
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    return x_adv


def corruption_uniform(model, x, y, epsilon=8/255, attack=False):
    x_adv = sample_delta(x, epsilon)
    if attack:
        return x_adv
    else:
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        return loss, logits


def corruption_gaussian(model, x, y, epsilon=8/255, attack=False):
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.clamp(torch.randn_like(x_adv), -epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()
 
    if attack:
        return x_adv
    else:
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        return loss, logits



def corruption_laplace(model, x, y, epsilon=8/255, attack=False):
    lap = dist.Laplace(loc=torch.tensor(0.0, device=x.device), scale=torch.tensor(epsilon, device=x.device))

    x_adv = x.detach().clone()
    x_adv = x_adv + torch.clamp(lap.sample(x_adv.shape).to(x.device), -epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()
 
    if attack:
        return x_adv
    else:
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        return loss, logits


def ERM_DataAug(model, x, y, epsilon=8/255, sample_num = 20):
    loss = 0 
    for _ in range(sample_num):
        x_adv = sample_delta(x, epsilon)
        logits = model(x_adv)
        loss += F.cross_entropy(logits, y, reduction="mean")

    loss = loss / float(sample_num)
    return loss, logits



def fgsm_loss(model, x, y, epsilon=8/255):
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()

    x_adv.requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y, reduction="mean")
    grad = torch.autograd.grad(loss, [x_adv])[0]
    x_adv = x + epsilon * grad.sign()
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()

    logits = model(x_adv)
    loss = F.cross_entropy(logits, y, reduction="mean")
    return loss, logits


def pgd_loss(model, x, y, optimizer, step_size=2/255, epsilon=8/255, attack_steps=10, attack=False):
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    for _ in range(attack_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv) 
        loss = F.cross_entropy(logits, y, reduction="mean")
        optimizer.zero_grad()
        loss.backward()                      
        grad = x_adv.grad.detach().sign()
        x_adv = x_adv + step_size * grad
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        
    if attack:
        return x_adv
    else:
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        return loss, logits


def pgd_origin(model, x, y, optimizer, step_size=2/255, epsilon=8/255, attack_steps=10, attack=False):
    x_adv = x.detach().clone()
    for _ in range(attack_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv) 
        loss = F.cross_entropy(logits, y, reduction="mean")
        optimizer.zero_grad()
        loss.backward()                      
        grad = x_adv.grad.detach().sign()
        x_adv = x_adv + step_size * grad
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        
    if attack:
        return x_adv
    else:
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        return loss, logits


def KL_AE(model, x, step_size, epsilon, attack_steps):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.eval()
    x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
    for _ in range(attack_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                    F.softmax(model(x), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    
    return x_adv


def trades_loss(model, x, y, optimizer, step_size=2/255, epsilon=8/255, attack_steps=10, beta=6.0):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    x_adv = KL_AE(model, x, step_size=step_size, epsilon=epsilon, attack_steps=attack_steps)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    logits = model(x)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                               F.softmax(model(x), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, logits


def mart_loss(model,x, y, optimizer, step_size=2/255, epsilon=8/255, attack_steps=10, beta=5.0):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x)
    x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
    for _ in range(attack_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    logits = model(x)
    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:] 
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust
    return loss, logits


def CVaR_loss(model, x, y, optimizer, epsilon=8/255, t_step_size=1.0, attack_steps=5, beta=0.5, M=20):
    batch_size = x.shape[0]
    ts = torch.ones(batch_size, device=x.device) 

    def sample_deltas(x, epsilon):
        return 2 * epsilon * torch.rand_like(x) - epsilon

    optimizer.zero_grad()
    for _ in range(attack_steps):

        cvar_loss, indicator_sum = 0, 0
        for _ in range(M):  
            perturbed_x = torch.clamp(x + sample_deltas(x, epsilon), 0, 1) 
            logits = model(perturbed_x)
            curr_loss = F.cross_entropy(logits, y, reduction='none')  
            indicator_sum += torch.where(curr_loss > ts, torch.ones_like(ts), torch.zeros_like(ts))
            cvar_loss += F.relu(curr_loss - ts) 

        indicator_avg = indicator_sum / M
        cvar_loss = (ts + cvar_loss / (M * beta)).mean()

        grad_ts = (1 - (1 / beta) * indicator_avg) / batch_size
        ts = ts - t_step_size * grad_ts 
        
    return cvar_loss, logits




def PR(model, x, y, step_size=2/255, epsilon=8/255, attack_steps=10):
    x_adv_list = []
    pgd = pgd_loss(model, x, y, step_size=step_size, epsilon=epsilon, attack_steps=attack_steps, attack=True)
    x_adv_list.append(pgd)
    while len(x_adv_list) < 5:
        epilon = random.uniform(epsilon - 0.02, epsilon)
        alpha = random.uniform(step_size  - 0.003, step_size + 0.003)
        num_iter = random.randint(attack_steps - 2, attack_steps + 5)
        x_adv = pgd_loss(model, x, y, step_size=alpha, epsilon=epilon, attack_steps=num_iter, attack=True)
        x_adv_list.append(x_adv)


    final_pr = pick_best_ae(step_size, model, x, x_adv_list, y)
    logits = model(final_pr)
    loss = F.cross_entropy(logits, y, reduction="mean")
    return loss, logits


def pick_best_ae(step_size, model, x, adv_list, y):
    max_distance = torch.zeros(y.size(0)).cuda()
    final_adv_example = adv_list[0] if adv_list else x
    for x_adv in adv_list:
        x_curr = x_adv.clone().detach()
        refine_lr = step_size
        x_curr = x_curr.requires_grad_(True)
        model.zero_grad()
        logits = model(x_curr)
        pred = logits.argmax(dim=1)
        is_ae = pred != y
        count = 0
        while is_ae.sum() > y.size(0) * 0.1:
            if count >= 70:
                break
            count += 1
            loss = F.cross_entropy(logits, y, reduction="mean")
            model.zero_grad()
            loss.backward()
            grad = x_curr.grad.detach()

            x_curr.data[is_ae] = x_curr.data[is_ae] - refine_lr * grad.data[is_ae].sign()
            x_curr.data[is_ae] = torch.clamp(x_curr.data[is_ae], 0, 1)

            x_curr = x_curr.detach().clone().requires_grad_(True)
            logits = model(x_curr)
            pred = logits.argmax(dim=1)
            is_ae = pred != y

        distance = torch.norm((x_adv - x_curr).view(x_adv.size(0), -1), dim=1, p=float('inf'))
        final_adv_example[distance>max_distance] = x_adv[distance>max_distance]
        max_distance[distance>max_distance] = distance[distance>max_distance]
    return final_adv_example



def TERM(model, x, y, t=2.0):
    logits = model(x) 
    loss = F.cross_entropy(logits, y, reduction="none")  
    term_loss = torch.log(torch.exp(t * loss).mean() + 1e-6) / t  
    return term_loss, logits


def ALP(model, x, y, optimizer, step_size=2/255, epsilon=8/255, attack_steps=10):
    x_pgd = pgd_loss(model, x, y, optimizer=optimizer,
                         step_size=step_size, 
                         epsilon=epsilon, 
                         attack_steps=attack_steps, 
                         attack=True)
    optimizer.zero_grad()
    logits = model(x_pgd) 
    robust_loss = F.cross_entropy(logits, y, reduction="mean")
    logit_diff = model(x_pgd) - model(x)
    logit_pairing_loss = torch.norm(logit_diff, dim=1).mean()
    total_loss = robust_loss + 0.5 * logit_pairing_loss
    return total_loss, logits


def CLP(model, x, y, optimizer, step_size=2/255, epsilon=8/255, attack_steps=10):
    x_pgd = pgd_loss(model, x, y, optimizer=optimizer,
                         step_size=step_size, 
                         epsilon=epsilon, 
                         attack_steps=attack_steps, 
                         attack=True)
    optimizer.zero_grad()
    logits = model(x) 
    clean_loss = F.cross_entropy(logits, y, reduction="mean")
    logit_diff = model(x_pgd) - model(x)
    logit_pairing_loss = torch.norm(logit_diff, dim=1).mean()
    total_loss = clean_loss +  0.3 * logit_pairing_loss
    return total_loss, logits





def shape_function(x, kind="linear"):
    if kind == "linear":
        return x
    elif kind == "softmax":
        return F.softmax(x, dim=0)
    elif kind == "exp":
        return torch.exp(x)
    else:
        raise ValueError(f"Unsupported shape function: {kind}")

def update_parameters(mu, sigma, model, x, y, epsilon, zeta, alpha, shape="linear", M=10):
    B = x.size(0)
    device = x.device

    u = torch.randn(M, B, *x.shape[1:], device=device)
    delta = epsilon * torch.tanh(u)
    x_perturbed = torch.clamp(x.unsqueeze(0) + delta, 0, 1)  # [M, B, C, H, W]
    x_perturbed_flat = x_perturbed.view(-1, *x.shape[1:])
    y_repeat = y.repeat(M)

    logits = model(x_perturbed_flat)
    losses = F.cross_entropy(logits, y_repeat, reduction="none").view(M, B)

    F_i = torch.exp(zeta * losses).mean(dim=0)  # [B]
    F_i_norm = (F_i - F_i.min()) / (F_i.max() - F_i.min() + 1e-8)
    F_i_shaped = shape_function(F_i_norm, shape).detach()

    delta_mean = delta.mean(dim=0)  # [B, C, H, W]
    delta_centered = delta - delta_mean.unsqueeze(0)
    weighted_diff = (F_i_shaped.view(1, B, 1, 1, 1) * delta_centered).sum(dim=1)  # [M, C, H, W]

    mu_new = mu + alpha * weighted_diff.mean(dim=0)
    sigma_new = torch.sqrt(
        ((F_i_shaped.view(1, B, 1, 1, 1) * delta_centered ** 2).sum(dim=1).mean(dim=0)) + 1e-6
    )

    return mu_new.detach(), sigma_new.detach()


def evar_risk_averse_step(model, x, y, mu, sigma, gamma=0.05, epsilon=0.1,
                          K=5, alpha=0.01, alpha_zeta=0.1, shape="linear", M=10, zeta_init=10.0):
    device = x.device
    B = x.size(0)

    zeta = torch.tensor(zeta_init, device=device, requires_grad=True)

    for k in range(K):
        mu, sigma = update_parameters(mu, sigma, model, x, y, epsilon, zeta, alpha, shape, M)

    # Final perturbation and EVaR computation
    u_final = torch.randn(M, B, *x.shape[1:], device=device)
    delta_final = epsilon * torch.tanh(u_final)
    x_perturbed = torch.clamp(x.unsqueeze(0) + delta_final, 0, 1).view(-1, *x.shape[1:])
    y_repeat = y.repeat(M)

    logits = model(x_perturbed)
    losses = F.cross_entropy(logits, y_repeat, reduction="none").view(M, B)

    exp_loss = torch.exp(zeta * losses)
    evar = (1 / zeta) * torch.log(exp_loss.mean(dim=0) / gamma)
    evar_mean = evar.mean()

    zeta_grad = torch.autograd.grad(evar_mean, zeta, retain_graph=True)[0]
    with torch.no_grad():
        new_zeta = zeta - alpha_zeta * zeta_grad
        new_zeta = new_zeta.clamp(min=1e-3, max=100.0)
        zeta = new_zeta.detach().clone().requires_grad_(True)

    mean_logits = logits.view(M, B, -1).mean(dim=0)
    return evar_mean, mean_logits
  
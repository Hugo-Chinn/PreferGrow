import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import graph_lib
from model import utils as mutils


def get_loss_fn(noise, graph, train, loss_type="score_entropy", sampling_eps=1e-3, lv=False):

    def loss_fn(model, batch, steps, cond=None, t=None, perturbed_target=None):
        """
        seq shape: [B, L] 
        target shape: [B]
        """
        history = batch["seq"]
        len_seq = batch["len_seq"]
        target = batch["next"]

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                #t = (1 - sampling_eps) * torch.rand(target.shape[0], device=target.device) + sampling_eps # [B]
                # 1) 生成离散的 timesteps
                t = torch.linspace(1.0, sampling_eps, steps + 1, device=target.device)[ \
                    torch.randint(0, steps + 1, (target.shape[0],), device=target.device)]
            
        sigma, dsigma = noise(t) # [B], [B]
        
        if perturbed_target is None:
            if len(target.size()) == 1:
                perturbed_target = graph.sample_prob(target[:,None], sigma[:, None])
            elif len(target.size()) == 2:
                perturbed_target = graph.sample_prob(target, sigma[:, None])

        log_score_fn = mutils.get_score_fn(model, None, train=True, sampling=False)
        
        log_score = log_score_fn(history, perturbed_target, sigma)
        if loss_type == "score_entropy":
            #print(log_score.shape)
            #print(sigma[:, None].shape)
            #print(perturbed_target.shape)
            #print(target.shape)
            loss = graph.score_entropy(log_score, sigma[:, None], perturbed_target, target)
            loss = (dsigma[:, None] * loss).mean(dim=-1)
        elif loss_type == "score_entropy_raw":
            loss = graph.score_entropy(log_score, sigma[:, None], perturbed_target, target)
            loss = loss.mean(dim=-1)
        elif loss_type == "score_mse":
            loss = graph.score_mse(log_score, sigma[:, None], perturbed_target, target)
            loss = loss.sum(dim=-1)
        elif loss_type == "score_se":
            loss = graph.score_mse(log_score, sigma[:, None], perturbed_target, target)
            loss = loss.sum(dim=-1)

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, loss_type, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train, loss_type)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, steps, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, steps, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, steps, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn
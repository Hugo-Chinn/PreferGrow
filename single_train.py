import numpy as np
import pandas as pd
import math
import random
import argparse
import datetime
import time
import os
import os.path
import gc
import logging
from itertools import chain

import hydra
from omegaconf import DictConfig

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

### load_dataloader
import data
### traing
import losses
### inference
import sampling

import graph_lib
import noise_lib
import utils
from model.transformer import SEDD4REC
from model.ema import ExponentialMovingAverage
from load_model import load_model
###TBD from utility import evaluate

torch.backends.cudnn.benchmark = True

logging.getLogger().setLevel(logging.INFO)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    #args = parse_args()
    setup_seed(cfg.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda)
    
    #torch.cuda.set_device(rank)
    work_dir = cfg.work_dir
    checkpoint_dir = './checkpoints/' + cfg.training.data +"/"
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta",cfg.training.data)

    print(work_dir)
    print(cfg)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    
    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    score_model = SEDD4REC(cfg).to(device)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    print(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    print(score_model)
    print(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    print(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    print(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 
    initial_step = int(state['step'])

    train_loader, val_loader, test_loader = data.get_seqdataloader(cfg)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    print(f"Length of datasets: {len(train_loader)}, {len(val_loader)}, {len(test_loader)}")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, cfg.loss_type, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, cfg.loss_type, optimize_fn, cfg.training.accum)


    if cfg.training.snapshot_sampling:
        #sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        #sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), 1)
        #sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)
        sampling_fn_0 = sampling.get_sampling_fn(cfg, graph, noise, sampling_eps, 1, device)
        sampling_fn_2 = sampling.get_sampling_fn(cfg, graph, noise, sampling_eps, cfg.sampling.personalization_strength, device)
        sampling_fn_5 = sampling.get_sampling_fn(cfg, graph, noise, sampling_eps, 5, device)
        sampling_fn_10 = sampling.get_sampling_fn(cfg, graph, noise, sampling_eps, 10, device)
        #one_step_score = sampling.get_eval_score(cfg, graph, noise, device)

    num_train_steps = cfg.training.n_iters
    print(f"Starting training loop at step {initial_step}.")
    
    start_time = time.time()

    #_,_ = utils.evaluate_loader(score_model, sampling_fn_0, val_loader, device)
    _,_ = utils.evaluate_loader(score_model, sampling_fn_2, val_loader, device)

    best_ndcg10 = 0  # 初始化最佳验证集指标为0
    patience = 50  # 设置容忍次数
    counter = 0  # 计数器
    epochs = 0
    T = 0.0

    # while epochs < 2000:
    #     for batch in train_loader:
    #         step = state['step']
    #         batch = {key: value.to(device) for key, value in batch.items()}
    #         loss = train_step_fn(state, batch, cfg.sampling.steps)
    #         #epochs += 1
    #         # flag to see if there was movement ie a full batch got computed
    #         if step != state['step']:
    #             current_time = time.time()
    #     epochs += 1
            
    #     if epochs % 1 == 0:
                
    #         elapsed_time = current_time - start_time
    #         print("epoch: %d, step: %d, training_loss: %.5e, time_elapsed: %.2fs" % (epochs, step, loss.item(), elapsed_time))
    #         start_time = current_time
            
    #         # if step % cfg.training.snapshot_freq_for_preemption == 0:
    #         #     utils.save_single_checkpoint(checkpoint_meta_dir, state)

    #         if epochs % 1 == 0:
    #             try:
    #                 eval_batch = next(val_iter)
    #             except StopIteration:
    #                 val_iter = iter(val_loader)  
    #                 eval_batch = next(val_iter)  
    #             eval_batch = {key: value.to(device) for key, value in eval_batch.items()}
    #             eval_loss = eval_step_fn(state, eval_batch, cfg.sampling.steps)
                
    #             eval_elapsed_time = time.time() - current_time
    #             print("step: %d, evaluation_loss: %.5e, eval_time: %.2fs" % (step, eval_loss.item(), eval_elapsed_time))

    #         if step > 0 and epochs % 5 == 0 or step == num_train_steps:
    #             # Save the checkpoint.
    #             save_step = step // cfg.training.snapshot_freq
    #             utils.save_single_checkpoint(os.path.join(
    #                 checkpoint_dir, f'checkpoint_{cfg.graph.type}_{save_step}.pth'), state)
                
    #             # Generate and save samples
    #             if cfg.training.snapshot_sampling:
    #                 print(f"Generating items at step: {step}")
                    
    #                 ema.store(score_model.parameters())
    #                 ema.copy_to(score_model.parameters())
    #                 print(f"without personalzation strength")
    #                 _,_ = utils.evaluate_loader(score_model, sampling_fn_0, val_loader, device)
    #                 print(f"with personalzation strength 2")
    #                 _,nd_list = utils.evaluate_loader(score_model, sampling_fn_2, val_loader, device)
    #                 # print(f"with personalzation strength 5")
    #                 # _,_ = utils.evaluate_loader(score_model, sampling_fn_5, val_loader, device)
    #                 # print(f"with personalzation strength 10")
    #                 # _,_ = utils.evaluate_loader(score_model, sampling_fn_10, val_loader, device)
    #                 print("test phase:")
    #                 print(f"without personalzation strength")
    #                 _,_ = utils.evaluate_loader(score_model, sampling_fn_0, test_loader, device)
    #                 print(f"with personalzation strength 2")
    #                 _,_ = utils.evaluate_loader(score_model, sampling_fn_2, test_loader, device)
    #                 # print(f"with personalzation strength 5")
    #                 # _,_ = utils.evaluate_loader(score_model, sampling_fn_5, test_loader, device)
    #                 # print(f"with personalzation strength 10")
    #                 # _,_ = utils.evaluate_loader(score_model, sampling_fn_10, test_loader, device)
    #                 ema.restore(score_model.parameters())

    #                 tv_ndcg10 = nd_list[2]
    #                 if tv_ndcg10 > best_ndcg10:
    #                     best_ndcg10 = tv_ndcg10
    #                     counter = 0  # 重置计数器
    #                     # 保存模型
    #                     print("\n best NDCG@20 is updated to ",best_ndcg10,"at epoch", epochs)
    #                     utils.save_single_checkpoint(os.path.join(
    #                         checkpoint_meta_dir, f'checkpoint_{cfg.graph.type}.pth'), state)
    #                 else:
    #                     counter += 1
    #                 if counter >= patience:
    #                     model, graph, noise = load_model("./", device)
    #                     for i in range(21):
    #                         sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_eps, i, device)
    #                         _,_ = utils.evaluate_loader(score_model, sampling_fn, test_loader, device)
    #                     break   # 停止训练循环
    
    while state['step'] < num_train_steps:
        for batch in train_loader:
            step = state['step']
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = train_step_fn(state, batch, cfg.sampling.steps)
            #epochs += 1
            # flag to see if there was movement ie a full batch got computed
            if step != state['step']:
                current_time = time.time()
            
                if step % cfg.training.log_freq == 0:
                
                    elapsed_time = current_time - start_time
                    print("step: %d, training_loss: %.5e, time_elapsed: %.2fs" % (step, loss.item(), elapsed_time))
                    start_time = current_time
            
                if step % cfg.training.snapshot_freq_for_preemption == 0:
                    save_step = step // cfg.training.snapshot_freq
                    utils.save_single_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{cfg.graph.type}_{save_step}.pth'), state)
                    #utils.save_single_checkpoint(checkpoint_meta_dir, state)

                if step % cfg.training.eval_freq == 0:
                    try:
                        eval_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)  
                        eval_batch = next(val_iter)  
                    eval_batch = {key: value.to(device) for key, value in eval_batch.items()}
                    eval_loss = eval_step_fn(state, eval_batch, cfg.sampling.steps)
                
                    eval_elapsed_time = time.time() - current_time
                    print("step: %d, evaluation_loss: %.5e, eval_time: %.2fs" % (step, eval_loss.item(), eval_elapsed_time))

                if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                    # Save the checkpoint.
                    save_step = step // cfg.training.snapshot_freq
                    utils.save_single_checkpoint(os.path.join(
                            checkpoint_meta_dir, f'checkpoint_{cfg.graph.type}.pth'), state)
                    #utils.save_single_checkpoint(os.path.join(
                    #    checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
                
                    # Generate and save samples
                    if cfg.training.snapshot_sampling:
                        print(f"Generating items at step: {step}")
                    
                        ema.store(score_model.parameters())
                        ema.copy_to(score_model.parameters())
                        #             print(f"without personalzation strength")
                        _,_ = utils.evaluate_loader(score_model, sampling_fn_0, val_loader, device)
                        print(f"with personalzation strength 2")
                        _,_ = utils.evaluate_loader(score_model, sampling_fn_2, val_loader, device)
                        print(f"with personalzation strength 5")
                        _,_ = utils.evaluate_loader(score_model, sampling_fn_5, val_loader, device)
                        print(f"with personalzation strength 10")
                        _,_ = utils.evaluate_loader(score_model, sampling_fn_10, val_loader, device)
                        print("test phase:")
                        print(f"without personalzation strength")
                        _,_ = utils.evaluate_loader(score_model, sampling_fn_0, test_loader, device)
                        print(f"with personalzation strength 2")
                        _,_ = utils.evaluate_loader(score_model, sampling_fn_2, test_loader, device)
                        print(f"with personalzation strength 5")
                        _,_ = utils.evaluate_loader(score_model, sampling_fn_5, test_loader, device)
                        print(f"with personalzation strength 10")
                        _,_ = utils.evaluate_loader(score_model, sampling_fn_10, test_loader, device)

                        #_,_ = utils.evaluate_loader(score_model, sampling_fn, test_loader, device)
                        #sample = sampling_fn(score_model, history)
                        ema.restore(score_model.parameters())

                        #sentences = tokenizer.batch_decode(sample)
        #  # flag to see if there was movement ie a full batch got computed
        # if step != state['step']:
        #     current_time = time.time()
            
        #     if step % cfg.training.log_freq == 0:
                
        #         elapsed_time = current_time - start_time
        #         print("step: %d, training_loss: %.5e, time_elapsed: %.2fs" % (epochs, loss.item(), elapsed_time))
        #         start_time = current_time
            
        #     if step % cfg.training.snapshot_freq_for_preemption == 0:
        #         utils.save_single_checkpoint(checkpoint_meta_dir, state)

        #     if step % cfg.training.eval_freq == 0:
        #         try:
        #             eval_batch = next(val_iter)
        #         except StopIteration:
        #             val_iter = iter(val_loader)  
        #             eval_batch = next(val_iter)  
        #         eval_batch = {key: value.to(device) for key, value in eval_batch.items()}
        #         eval_loss = eval_step_fn(state, eval_batch, cfg.sampling.steps)
                
        #         eval_elapsed_time = time.time() - current_time
        #         print("step: %d, evaluation_loss: %.5e, eval_time: %.2fs" % (epochs, eval_loss.item(), eval_elapsed_time))

        #     if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
        #         # Save the checkpoint.
        #         save_step = epochs // cfg.training.snapshot_freq
        #         utils.save_single_checkpoint(os.path.join(
        #             checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
                
        #         # Generate and save samples
        #         if cfg.training.snapshot_sampling:
        #             print(f"Generating items at step: {step}")
                    
        #             ema.store(score_model.parameters())
        #             ema.copy_to(score_model.parameters())
        #             #_,_ = utils.evaluate_toy(score_model, sampling_fn, val_loader, device)
        #             print(f"without personalzation strength")
        #             _,_ = utils.evaluate_loader(score_model, sampling_fn_0, val_loader, device)
        #             print(f"with personalzation strength 2")
        #             _,_ = utils.evaluate_loader(score_model, sampling_fn_2, val_loader, device)
        #             print(f"with personalzation strength 5")
        #             _,NDCG = utils.evaluate_loader(score_model, sampling_fn_5, val_loader, device)
        #             print(f"with personalzation strength 10")
        #             _,_ = utils.evaluate_loader(score_model, sampling_fn_10, val_loader, device)
        #             print("test phase:")
        #             print(f"without personalzation strength")
        #             _,_ = utils.evaluate_loader(score_model, sampling_fn_0, test_loader, device)
        #             print(f"with personalzation strength 2")
        #             _,_ = utils.evaluate_loader(score_model, sampling_fn_2, test_loader, device)
        #             print(f"with personalzation strength 5")
        #             _,_ = utils.evaluate_loader(score_model, sampling_fn_5, test_loader, device)
        #             print(f"with personalzation strength 10")
        #             _,_ = utils.evaluate_loader(score_model, sampling_fn_10, test_loader, device)

        #             #_,_ = utils.evaluate_loader(score_model, sampling_fn, test_loader, device)
        #             #sample = sampling_fn(score_model, history)
        #             ema.restore(score_model.parameters())

        #             tv_ndcg20 = NDCG[3]
        #             if tv_ndcg20 > best_ndcg20:
        #                 best_ndcg20 = tv_ndcg20
        #                 counter = 0  # 重置计数器
        #                 # 保存模型
        #                 print("\n best NDCG@20 is updated to ",best_ndcg20,"at epoch", state['step'])
        #                 save_step = epochs // cfg.training.snapshot_freq
        #                 utils.save_single_checkpoint(os.path.join(
        #                     checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
        #             else:
        #                 counter += 1
        #             if counter >= patience:
        #                 break   # 停止训练循环
        #             print('----------------------------------------------------------------')

        #             #sentences = tokenizer.batch_decode(sample)
                    
if __name__ == "__main__":
    main()
import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from omegaconf import OmegaConf, open_dict


def load_hydra_config_from_run(load_dir):
    cfg_path = os.path.join(load_dir, "configs/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].module.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)
    
def save_single_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)
    
def calculate_hit_loader(sorted_list,topk,true_items,hit_purchase,ndcg_purchase,mrr_purchase):
    true_items = true_items.tolist()
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        # print(rec_list)
        # print(true_items)
        # print('...........')
        # break
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])[0,0]
                # total_reward[i] += rewards[j]
                # if rewards[j] == r_click:
                #     hit_click[i] += 1.0
                #     ndcg_click[i] += 1.0 / np.log2(rank + 1)
                # else:
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)
                mrr_purchase[i] += 1.0/ rank
                
def evaluate_loader(model, sampling_fn, data_loader, device):

    total_purchase = 0.0
    hit_purchase=[0,0,0,0,0]
    ndcg_purchase=[0,0,0,0,0]
    mrr_purchase = [0,0,0,0,0]
    topk = [1,5,10,20,50]

    for batch in data_loader:
        history = batch["seq"].to(device)
        target = batch["next"]
        
        if len(history) < 256:
            continue

        prediction = sampling_fn(model, (target.shape[0],1), history) 
        prediction = prediction[:,0,:]
        _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        sorted_list2=np.flip(topK,axis=1)
        calculate_hit_loader(sorted_list2,topk,target,hit_purchase,ndcg_purchase,mrr_purchase)
        total_purchase+=len(history)

    hr_list = []
    ndcg_list = []
    mrr_list = []
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        mr_purchase=mrr_purchase[i]/total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase)
        mrr_list.append(mr_purchase)
    print('{:<10s} {:<10s} {:<10s} '.format("ACC","ACC","ACC"))
    print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), mrr_list[0]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[1]), 'HR@'+str(topk[2]), 'HR@'+str(topk[3]), 'HR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], (hr_list[2]), hr_list[3], hr_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@'+str(topk[1]), 'NDCG@'+str(topk[2]), 'NDCG@'+str(topk[3]), 'NDCG@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], (ndcg_list[2]), ndcg_list[3], ndcg_list[4]))
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@'+str(topk[1]), 'MRR@'+str(topk[2]), 'MRR@'+str(topk[3]), 'MRR@'+str(topk[4])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], (mrr_list[2]), mrr_list[3], mrr_list[4]))

    return hr_list, ndcg_list

def evaluate_sample_KL(model, sampling_fn, data_loader, device):

    batch_num = 0
    KL_list = []
    for batch in data_loader:
        history = batch["seq"].to(device)
        target = batch["next"]
        if len(history) < 256:
            continue
        batch_num += 1

        transition_probs = sampling_fn(model, (target.shape[0],1), history) 
        if len(KL_list) == 0:
            KL_list = [ 0.0 for i in range(len(transition_probs))]
        for i, probs in enumerate(transition_probs):
            print(probs)
            probs_normalized = F.softmax(probs, dim=-1)  # (256, 12102)，归一化为概率
            batch_indices = torch.arange(probs.size(0), device=probs.device)  # [0,1,2,...,255]
    
            selected_probs = probs_normalized[batch_indices, target]  # shape: (256,)

            print(selected_probs)
    
            # 计算 -log(prob)
            neg_log_probs = -torch.log(selected_probs + 1e-9)  # 避免 log(0) 出现数值错误
    
            KL_list[i] += neg_log_probs.mean()  # 可以append，也可以求均值后append
    KL_list = [ KL_list[i] / batch_num for i in range(len(transition_probs))]
    

    return KL_list

def evaluate_toy(model, sampling_fn, data_loader, device):

    topk = [1, 5, 10, 20, 50]
    hit_purchase = [0] * len(topk)
    ndcg_purchase = [0] * len(topk)
    mrr_purchase = [0] * len(topk)

    # -------- 只取一个 batch --------
    batch = next(iter(data_loader))

    history = batch["seq"][:5].to(device)  # 只取前10个
    target = batch["next"][:5]
    prediction = sampling_fn(model, (5, 1), history).squeeze()

    _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
    topK = topK.cpu().detach().numpy()
    sorted_list2 = np.flip(topK, axis=1)

    calculate_hit_loader(sorted_list2, topk, target, hit_purchase, ndcg_purchase, mrr_purchase)
    
    total_purchase = 5  # =10

    hr_list = [h / total_purchase for h in hit_purchase]
    ndcg_list = [n / total_purchase for n in ndcg_purchase]
    mrr_list = [m / total_purchase for m in mrr_purchase]

    print(f"Batch Size = {total_purchase}")

    print('{:<10s} {:<10s} {:<10s}'.format("HR@1", "NDCG@1", "MRR@1"))
    print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], ndcg_list[0], mrr_list[0]))

    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@5', 'HR@10', 'HR@20', 'HR@50'))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], hr_list[2], hr_list[3], hr_list[4]))

    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50'))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], ndcg_list[2], ndcg_list[3], ndcg_list[4]))

    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@5', 'MRR@10', 'MRR@20', 'MRR@50'))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], mrr_list[2], mrr_list[3], mrr_list[4]))

    return hr_list, ndcg_list  # 可以按需改成你要用的指标

# def evaluate_preference_score(model, sampling_fn, data_loader, steps, device):

#     #for i in range(steps):
#     for batch in data_loader:
#         history = batch["seq"].to(device)
#         target = batch["next"]
        
#         if len(history) < 256:
#             continue

#         prediction = sampling_fn(model, (target.shape[0],1), history) 
#         prediction = prediction[:,0,:]
#         _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
#         topK = topK.cpu().detach().numpy()
#         sorted_list2=np.flip(topK,axis=1)
#         calculate_hit_loader(sorted_list2,topk,target,hit_purchase,ndcg_purchase,mrr_purchase)
#         total_purchase+=len(history)

#     hr_list = []
#     ndcg_list = []
#     mrr_list = []
#     for i in range(len(topk)):
#         hr_purchase=hit_purchase[i]/total_purchase
#         ng_purchase=ndcg_purchase[i]/total_purchase
#         mr_purchase=mrr_purchase[i]/total_purchase
#         hr_list.append(hr_purchase)
#         ndcg_list.append(ng_purchase)
#         mrr_list.append(mr_purchase)
#     print('{:<10s} {:<10s} {:<10s} '.format("ACC","ACC","ACC"))
#     print('{:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), mrr_list[0]))
#     print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[1]), 'HR@'+str(topk[2]), 'HR@'+str(topk[3]), 'HR@'+str(topk[4])))
#     print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[1], (hr_list[2]), hr_list[3], hr_list[4]))
#     print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('NDCG@'+str(topk[1]), 'NDCG@'+str(topk[2]), 'NDCG@'+str(topk[3]), 'NDCG@'+str(topk[4])))
#     print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(ndcg_list[1], (ndcg_list[2]), ndcg_list[3], ndcg_list[4]))
#     print('{:<10s} {:<10s} {:<10s} {:<10s}'.format('MRR@'+str(topk[1]), 'MRR@'+str(topk[2]), 'MRR@'+str(topk[3]), 'MRR@'+str(topk[4])))
#     print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(mrr_list[1], (mrr_list[2]), mrr_list[3], mrr_list[4]))

#     return hr_list, ndcg_list
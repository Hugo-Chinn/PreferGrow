import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


from catsample import sample_categorical

def get_graph(config, device):
    if config.training.data == "ATV":
        item_num = config.data.ATV.item_num
        #seq_len = config.data.ATV.seq_len
        #item_num = config.data.ATV.item_num
    elif config.training.data == "ML1M":
        item_num = config.data.ML1M.item_num
    elif config.training.data == "ATG":
        item_num = config.data.ATG.item_num
        #seq_len = config.data.ATV.seq_len
        #item_num = config.data.ATV.item_num
    elif config.training.data == "ASO":
        item_num = config.data.ASO.item_num
    elif config.training.data == "Beauty":
        item_num = config.data.Beauty.item_num
        #seq_len = config.data.ATV.seq_len
        #item_num = config.data.ATV.item_num
    elif config.training.data == "Steam":
        item_num = config.data.Steam.item_num

    if config.graph.type == "pair":
        return PairWise(item_num)
    elif config.graph.type == "point":
        return PointWise(item_num)
    elif config.graph.type == "hybrid":
        return HybridWise(item_num, config.graph.gamma)
    elif config.graph.type == "adaptive":
        return AdaptiveWise(item_num, config.graph.is_disliked_item)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class PreferGrow(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def is_disliked_item(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass


    @abc.abstractmethod
    def rate_matrix_col(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass


    @abc.abstractmethod
    def rate_matrix_row(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass


    @abc.abstractmethod
    def prob_matrix_col(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass


    def sample_prob(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")
    

    def reverse_prob(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    
    @abc.abstractmethod
    def reverse_prob_ratio(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass
    

    @abc.abstractmethod
    def sample_nonpreference(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass


    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass

"""
class PairWise(PreferGrow):
    """"""
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """"""
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim
    
    @property
    def is_disliked_item(self):
        return False

    def rate_matrix_col(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def rate_matrix_row(self, i):
        return self.rate(i)

    def prob_matrix_col(self, i, int_beta):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-int_beta[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans
    
    def prob_matrix_row(self, i, int_beta):
        return self.prob_matrix_col(i, int_beta)

    def sample_prob(self, i, int_beta):
        move_chance = 1. - (-int_beta).exp()
        # Move chance ratio (n-1)/n * (1. - (-sigma).exp())
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def reverse_prob_ratio(self, score, beta):
        """"""
        Computes the approximated ratios of P_{s} / P_{t}(x_t=i) as P_{t|s}^{-1} * exp_Rankings(x_t=i, t)
        P_{t|s}^{-1} = inverse_alpha * I + (1 - inverse_alpha) * E 
        """"""
        dim = score.shape[-1]
        epow = (-beta).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_nonpreference(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, int_beta, x, x0):
        #score = score.squeeze()
        esigm1 = torch.where(
            int_beta < 0.5,
            torch.expm1(int_beta),
            torch.exp(int_beta) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        # print(score.shape)
        # print(x0.shape)
        # print(x.shape)
        # print(esigm1.shape)
        # print(neg_term.shape)
        # neg_term = torch.where(
        #     x == x0,
        #     ratio * neg_term,
        #     torch.gather(score, -1, x0[..., None, None]).squeeze(-1) / esigm1 + neg_term
        # )
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None, None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        #positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return (pos_term - neg_term + const)


class PointWise(PreferGrow):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def is_disliked_item(self):
        return True

    def rate_matrix_col(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        

    def rate_matrix_row(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def prob_matrix_col(self, i, int_beta):
        B = i.shape[0]
        device = i.device
        alpha = (-int_beta[..., None]).exp()
        transition_prob_from_i = torch.zeros(B, self.dim, device=device)
        # (1 - alpha) * E 
        transition_prob_from_i[:, -1] = 1 - alpha
        #  alpha * I 
        transition_prob_from_i[torch.arange(B), i] += alpha
        return transition_prob_from_i
    
    def prob_matrix_row(self, i, int_beta):
        int_beta = unsqueeze_as(int_beta, i[..., None])
        edge = (-int_beta).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-int_beta).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_prob(self, i, int_beta):
        move_chance = 1 - (-int_beta).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert
    
    def reverse_prob_ratio(self, score, beta):
        """"""
        Computes the approximated ratios of P_{s} / P_{t}(x_t=i) as P_{t|s}^{-1} * exp_Rankings(x_t=i, t)
        P_{t|s}^{-1} = inverse_alpha * I + (1 - inverse_alpha) * E 
        """"""
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (beta).exp()) * score.sum(dim=-1)
        score *= beta.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_nonpreference(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, int_beta, x, x0):
        #print(score.shape)
        #print(sigma.shape)
        #print(x.shape)
        #print(x0.shape)
        x0 = x0[:, None]
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            int_beta < 0.5,
            torch.expm1(int_beta),
            torch.exp(int_beta) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += (pos_term - neg_term + const)
        return entropy
    
    def score_mse(self, score, sigma, x, x0):
        #print(score.shape)
        #print(sigma.shape)
        #print(x.shape)
        #print(x0.shape)
        x0 = x0[:, None]
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        # 创建 full-size ratio，行为根据 rel_ind 变化
        ratio = torch.where(rel_ind, 1. / esigm1.expand_as(x), esigm1.expand_as(x))
        # 计算 ratio 和 score 的每行平方差
        mse = ((score.exp() - ratio) ** 2).mean(dim=-1)  # shape: [batch_size]
        return mse

    def score_se(self, score, sigma, x, x0):
        #print(score.shape)
        #print(sigma.shape)
        #print(x.shape)
        #print(x0.shape)
        x0 = x0[:, None]
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        # 创建 full-size ratio，行为根据 rel_ind 变化
        ratio = torch.where(rel_ind, 1. / esigm1.expand_as(x), esigm1.expand_as(x))
        # 计算 ratio 和 score 的每行平方差
        mse = ((score.exp() - ratio) ** 2).sum(dim=-1)  # shape: [batch_size]
        return mse

class MixWise(PreferGrow):
    def __init__(self, dim, gamma):
        super().__init__()
        self._dim = dim
        self.gamma = gamma

    @property
    def dim(self):
        return self._dim + 1
    
    # absorb 2 a universally disliked item
    @property
    def is_disliked_item(self):
        return True

    # rate 2 rate_matrix_col without considering coefficients: E-I
    def rate_matrix_col(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        
    # rate 2 rate_matrix_row without considering coefficients: E-I
    def rate_matrix_row(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge
    # rate 2 prob_matrix_col: alpha_t
    def prob_matrix_col(self, i, int_beta):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-int_beta[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        pass
    
    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        #print(score.shape)
        #print(sigma.shape)
        #print(x.shape)
        #print(x0.shape)
        x0 = x0[:, None]
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy
    
    def score_mse(self, score, sigma, x, x0):
        #print(score.shape)
        #print(sigma.shape)
        #print(x.shape)
        #print(x0.shape)
        x0 = x0[:, None]
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        # 创建 full-size ratio，行为根据 rel_ind 变化
        ratio = torch.where(rel_ind, 1. / esigm1.expand_as(x), esigm1.expand_as(x))
        # 计算 ratio 和 score 的每行平方差
        mse = ((score.exp() - ratio) ** 2).mean(dim=-1)  # shape: [batch_size]
        return mse

    def score_se(self, score, sigma, x, x0):
        #print(score.shape)
        #print(sigma.shape)
        #print(x.shape)
        #print(x0.shape)
        x0 = x0[:, None]
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        # 创建 full-size ratio，行为根据 rel_ind 变化
        ratio = torch.where(rel_ind, 1. / esigm1.expand_as(x), esigm1.expand_as(x))
        # 计算 ratio 和 score 的每行平方差
        mse = ((score.exp() - ratio) ** 2).sum(dim=-1)  # shape: [batch_size]
        return mse
    
class HybridWise(PreferGrow):
    """"""
    E is a (dim + 1) dimensional rank-1 non-symmetric idempotent matrix:
    - first dim rows = gamma / dim
    - last row = 1 - gamma
    """"""
    def __init__(self, dim, gamma):
        super().__init__()
        self._dim = dim
        self.gamma = gamma  # gamma ∈ [0, 1]

    @property
    def dim(self):
        return self._dim + 1  # E is (dim+1)-dimensional

    @property
    def is_disliked_item(self):
        return True  # 被所有人厌恶的物品

    def rate_matrix_col(self, i):
        B = i.shape[0]
        edge = torch.full((B, self.dim), self.gamma / self._dim, device=i.device)
        edge[:, -1] = 1 - self.gamma
        edge = edge.scatter_add(-1, i[..., None], -torch.ones_like(i, dtype=edge.dtype)[..., None])
        return edge

    def rate_matrix_row(self, i):
        B = i.shape[0]
        device = i.device
        edge = torch.full((B, self.dim), self.gamma / self._dim, device=device)
        edge[i == self.dim - 1] = 1 - self.gamma
        edge = edge.scatter_add(-1, i[..., None], -torch.ones_like(i, dtype=edge.dtype)[..., None])
        return edge

    def prob_matrix_col(self, i, int_beta):
        B = i.shape[0]
        device = i.device
        alpha = (-int_beta[..., None]).exp()
        base = torch.full((B, self.dim), self.gamma / self._dim, device=device)
        base[:, -1] = 1 - self.gamma
        trans = (1 - alpha) * base
        trans.scatter_add_(-1, i[..., None], alpha)
        return trans

    def prob_matrix_row(self, i, int_beta):
        B = i.shape[0]
        device = i.device
        alpha = (-int_beta).exp()

        # Identity part: one-hot rows
        identity = F.one_hot(i, num_classes=self.dim).float()

        # E part: each row depends on whether i == dim - 1
        base = torch.full((B, self.dim), self.gamma / self._dim, device=device)
        base[i == self.dim - 1] = 1 - self.gamma

        # P = αI + (1 - α)E
        edge = alpha[:, None] * identity + (1 - alpha)[:, None] * base
        return edge

    def sample_prob(self, i, int_beta):
        # 移动的总概率（来自 int_beta）
        move_chance = 1 - (-int_beta).exp()
        move_mask = torch.rand_like(int_beta, dtype=torch.float, device=i.device) < move_chance

        # 针对“决定移动”的样本，使用 self.gamma 决定是去往 [0, dim-2] 还是 dim-1
        sample_gamma = torch.rand_like(i, dtype=torch.float, device=i.device)

        # 随机采样 [0, dim - 2] 的位置（当选择非最后一类时使用）
        non_final = torch.randint(0, self.dim - 1, i.shape, device=i.device)

        # 默认不移动
        i_pert = i.clone()

        # 根据 gamma 决定移动位置
        move_pairwise = (sample_gamma < self.gamma) & move_mask
        move_pointwise = (~move_pairwise) & move_mask

        i_pert[move_pairwise] = non_final[move_pairwise]
        i_pert[move_pointwise] = self.dim - 1

        return i_pert

    def reverse_prob_ratio(self, score, beta):
        """"""
        Computes the approximated ratios of P_{s} / P_{t}(x_t=i) as P_{t|s}^{-1} * exp_Rankings(x_t=i, t)
        P_{t|s}^{-1} = inverse_alpha * I + (1 - inverse_alpha) * E 
        """"""
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const1 = (1 - (beta).exp()) * (1 - self.gamma) * score.sum(dim=-1)
        extra_const2 = (1 - (beta).exp()) * self.gamma / self._dim * score.sum(dim=-1)
        score *= beta.exp()[:, None]
        score[..., :-1] += extra_const2
        score[..., -1] += extra_const1
        return score 

    def reverse_prob_ratio(self, score, beta):
        dim = score.shape[-1]
        epow = (-beta).exp()[..., None]

        E_row = torch.full_like(score, self.gamma / self._dim)
        E_row[..., -1] = 1 - self.gamma
        sum_score = (E_row * score).sum(dim=-1, keepdim=True)

        return ((epow - 1) * sum_score / dim) + score / epow

    def sample_nonpreference(self, *batch_dims):
        return torch.full(batch_dims, self.dim - 1, dtype=torch.int64)

    def sample_nonpreference(self, *batch_dims):
        # 随机采样，决定是否取最后一类（dim - 1）
        sample_mask = torch.rand(*batch_dims) < self.gamma
        # 在 [0, dim - 2] 范围内随机采样
        non_final = torch.randint(0, self.dim - 1, batch_dims)
        # 最终结果：gamma 概率为 dim - 1，剩下随机取其他类
        return torch.where(sample_mask, torch.full(batch_dims, self.dim - 1), non_final)
    
    def score_entropy(self, score, int_beta, x, x0):
        #score = score.squeeze()
        # esigm1 = (1-alpha_t) / alpha_t
        esigm1 = torch.where(
            int_beta < 0.5,
            torch.expm1(int_beta),
            torch.exp(int_beta) - 1
        )
        ratio_base = (1 - self.dim / ( self.gamma * esigm1 + self.dim))

        score_target = torch.gather(score, -1, x0[..., None])

        neg_term_base = (score.sum(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1)) / (self.dim - 1)
        # neg_term_like = self.gamma * neg_term_base + (1 - self.gamma * (1 + 1. / (self.dim - 1))) * score[...,-1]
        # neg_term_mild = self.gamma * neg_term_base + ( 1. / ratio_base - 1) * self.gamma / (self.dim - 1) * torch.gather(score, -1, x0[..., None]) + \
        #                 (1 - self.gamma  * (1 + 1. / (self.dim - 1))) * score[...,-1]
        # neg_term_hate = self.gamma * neg_term_base + ( 1. / ratio_base - 1) * self.gamma / (self.dim - 1) * torch.gather(score, -1, x0[..., None])
        # print(score.shape)
        # print(x0.shape)
        # print(x.shape)
        # print(esigm1.shape)
        # print(neg_term.shape)
        #neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim

        # neg_term = torch.where(
        #     x == x0,
        #     ratio_base * neg_term_like,
        #     torch.where(
        #         x == self.dim - 1,
        #         neg_term_hate,
        #         neg_term_mild
        #     )
        # )

        # neg_term = torch.where(
        #     x == x0,
        #     ratio_base * (self.gamma * neg_term_base + (1 - self.gamma * (1 + 1. / (self.dim - 1))) * score[...,-1]),
        #     torch.where(
        #         x == self.dim - 1,
        #         self.gamma * neg_term_base + ( 1. / ratio_base - 1) * self.gamma / (self.dim - 1) * torch.gather(score, -1, x0[..., None]),
        #         self.gamma * neg_term_base + ( 1. / ratio_base - 1) * self.gamma / (self.dim - 1) * torch.gather(score, -1, x0[..., None]) + \
        #                  (1 - self.gamma  * (1 + 1. / (self.dim - 1))) * score[...,-1]
        #     )
        # )
        # like 情况 (x == x0)
        neg_like = ratio_base * (
            self.gamma * neg_term_base +
            (1 - self.gamma * (1 + 1. / (self.dim - 1))) * score[..., -1]
        )
        # hate 情况 (x == dim - 1)
        neg_hate = self.gamma * neg_term_base + \
            (1. / ratio_base - 1) * self.gamma / (self.dim - 1) * score_target
        # mild 情况 (其他)
        neg_mild = self.gamma * neg_term_base + \
            (1. / ratio_base - 1) * self.gamma / (self.dim - 1) * score_target + \
            (1 - self.gamma * (1 + 1. / (self.dim - 1))) * score[..., -1]

        # 合并逻辑判断（矢量化）
        neg_term = torch.where(
            x == x0,
            neg_like,
            torch.where(
                x == self.dim - 1,
                neg_hate,
                neg_mild
            )
        )

        gamma_const = (self.gamma) / ((1 - self.gamma) * (self.dim - 1))

        # constant factor
        const = torch.where(
            x == x0,
            (1 - self.gamma / (self.dim - 1)) * ratio_base * (ratio_base.log() - 1.) - (1 - self.gamma) * ratio_base * gamma_const.log(),
            torch.where(
                x == self.dim - 1,
                self.gamma * (1 + (1. / ratio_base - 1.) / (self.dim - 1)) * (gamma_const.log() - 1.) - ratio_base.log() / ratio_base,
                -(1 - self.gamma) * (gamma_const.log() + 1.) - (ratio_base.log() + self.gamma.log() + 1.) / (ratio_base * (self.dim - 1)) - self.gamma * (1 - 2. / (self.dim - 1))
            )
        )

        #positive term
        sexp = score.exp()
        pos_term_base = (sexp.sum(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1)) / (self.dim - 1)
        pos_term = torch.where(
            x == self.dim - 1,
            (self.dim - 1) * (1 - self.gamma) * pos_term_base,
            self.gamma * pos_term_base
        )
        return pos_term - neg_term + const
"""

class PairWise(PreferGrow):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim
    
    @property
    def is_disliked_item(self):
        return False

    def rate_matrix_col(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def rate_matrix_row(self, i):
        return self.rate_matrix_col(i)

    def prob_matrix_col(self, i, int_beta):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-int_beta[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans
    
    def prob_matrix_row(self, i, int_beta):
        return self.prob_matrix_col(i, int_beta)

    def sample_prob(self, i, int_beta):
        move_chance = 1. - (-int_beta).exp()
        # Move chance ratio (n-1)/n * (1. - (-sigma).exp())
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert
    
    @torch.no_grad()
    def reverse_prob_ratio(self, exp_score, minus_beta):
        """
        Computes the approximated ratios of P_{s} / P_{t}(x_t=i) as P_{t|s}^{-1} * exp_Rankings(x_t=i, t)
        P_{t|s}^{-1} = inverse_alpha * I + (1 - inverse_alpha) * E 
        """
        dim = exp_score.shape[-1]
        epow = (-minus_beta).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * exp_score.sum(dim=-1, keepdim=True) + exp_score / epow

    def sample_nonpreference(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, int_beta, x, x0):
        x0 = x0.unsqueeze(-1)
        #score = score.squeeze()
        esigm1 = torch.where(
            int_beta < 0.5,
            torch.expm1(int_beta),
            torch.exp(int_beta) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        #print(score.shape)
        #print(ratio.shape)

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim

        #print("neg_term.shape =", neg_term.shape)
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        # neg_term = torch.where(
        #     x == x0,
        #     ratio * neg_term,
        #     torch.gather(score, -1, x0[..., None, None]).squeeze(-1) / esigm1 + neg_term
        # )
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term 
        )

        #print("neg_term.shape =", neg_term.shape)

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        #positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return (pos_term - neg_term + const)

class PointWise(PreferGrow):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def is_disliked_item(self):
        return True

    def rate_matrix_col(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        

    def rate_matrix_row(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def prob_matrix_col(self, i, int_beta):
        B = i.shape[0]
        device = i.device
        alpha = (-int_beta[..., None]).exp()
        transition_prob_from_i = torch.zeros(B, self.dim, device=device)
        # (1 - alpha) * E 
        transition_prob_from_i[:, -1] = 1 - alpha
        #  alpha * I 
        transition_prob_from_i[torch.arange(B), i] += alpha
        return transition_prob_from_i
    
    def prob_matrix_row(self, i, int_beta):
        int_beta = unsqueeze_as(int_beta, i[..., None])
        edge = (-int_beta).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-int_beta).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_prob(self, i, int_beta):
        move_chance = 1 - (-int_beta).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert
    
    @torch.no_grad()
    def reverse_prob_ratio(self, exp_score, minus_beta):
        """
        Computes the approximated ratios of P_{s} / P_{t}(x_t=i) as P_{t|s}^{-1} * exp_Rankings(x_t=i, t)
        P_{t|s}^{-1} = inverse_alpha * I + (1 - inverse_alpha) * E 
        """
        #score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (minus_beta).exp()) * exp_score.sum(dim=-1)
        exp_score *= minus_beta.exp()[:, None]
        exp_score[..., -1] += extra_const
        return exp_score

    def sample_nonpreference(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, int_beta, x, x0):

        x0 = x0[:, None]
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            int_beta < 0.5,
            torch.expm1(int_beta),
            torch.exp(int_beta) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += (pos_term - neg_term + const)
        return entropy
    
class HybridWise(PreferGrow):
    """
    E is a (dim + 1) dimensional rank-1 non-symmetric idempotent matrix:
    - first dim rows = gamma / dim
    - last row = 1 - gamma
    """
    def __init__(self, dim, gamma):
        super().__init__()
        self._dim = dim
        self.gamma = torch.tensor(gamma)  # gamma ∈ [0, 1]

    @property
    def dim(self):
        return self._dim + 1  # E is (dim+1)-dimensional

    @property
    def is_disliked_item(self):
        return True  # 被所有人厌恶的物品

    def rate_matrix_col(self, i):
        B = i.shape[0]
        edge = torch.full((B, self.dim), self.gamma / self._dim, device=i.device)
        edge[:, -1] = 1 - self.gamma
        edge = edge.scatter_add(-1, i[..., None], -torch.ones_like(i, dtype=edge.dtype)[..., None])
        return edge

    def rate_matrix_row(self, i):
        B = i.shape[0]
        device = i.device
        edge = torch.full((B, self.dim), self.gamma / self._dim, device=device)
        edge[i == self.dim - 1] = 1 - self.gamma
        edge = edge.scatter_add(-1, i[..., None], -torch.ones_like(i, dtype=edge.dtype)[..., None])
        return edge

    # def prob_matrix_col(self, i, int_beta):
    #     B = i.shape[0]
    #     device = i.device
    #     alpha = (-int_beta).exp()
    #     base = torch.full((B, self.dim), self.gamma / self._dim, device=device)
    #     base[:, -1] = 1 - self.gamma
    #     trans = (1 - alpha) * base
    #     trans[torch.arange(B), i] += alpha
    #     return trans

    # def prob_matrix_row(self, i, int_beta):
    #     B = i.shape[0]
    #     device = i.device
    #     alpha = (-int_beta).exp()

    #     # Identity part: one-hot rows
    #     identity = F.one_hot(i, num_classes=self.dim).float()

    #     # E part: each row depends on whether i == dim - 1
    #     print(i.shape)
    #     base = torch.full((B, self.dim), self.gamma / (self.dim - 1), device=device)
    #     base[i.squeeze() == self.dim - 1] = 1 - self.gamma

    #     # P = αI + (1 - α)E
    #     edge = alpha * identity + (1 - alpha) * base
    #     return edge

    def prob_matrix_col(self, i, int_beta):
        # B, L = i.shape
        alpha = (-int_beta).exp()                         # [B, 1]
        one_minus_alpha = (1 - alpha)                   # [B, 1]

        # 初始化 transition matrix: [B, 1, dim]
        trans = torch.ones(*i.shape, self.dim, device=i.device) * self.gamma * one_minus_alpha.unsqueeze(-1) / (self.dim - 1)
        # trans = torch.full((B, L, self.dim), fill_value=self.gamma * one_minus_alpha / (self.dim - 1), device=i.device)
        trans[..., -1] = (1 - self.gamma) * one_minus_alpha

        # scatter 加入 α 到对应位置
        trans = trans.scatter_add(-1, i.unsqueeze(-1), alpha.unsqueeze(-1))
        return trans  # [B, 1, dim]

    
    def prob_matrix_row(self, i, int_beta):
        # B, L = i.shape
        alpha = (-int_beta).exp()                         # [B, 1]
        one_minus_alpha = (1 - alpha)       # [B, 1, 1]

        # 初始化 transition matrix: [B, 1, dim]
        trans = torch.ones(*i.shape, self.dim, device=i.device) * self.gamma * one_minus_alpha.unsqueeze(-1) / (self.dim - 1)
        # trans = torch.full((B, L, self.dim), fill_value=self.gamma * one_minus_alpha / (self.dim - 1), device=i.device)
        # shape: [B, L, 1] → broadcast → [B, L, dim]
        hate_row = ((1 - self.gamma) * one_minus_alpha / self.dim).unsqueeze(-1).expand(-1, -1, self.dim)
        # 构造 mask: [B, L, 1] → broadcast → [B, L, dim]
        mask = (i == self.dim - 1).unsqueeze(-1).expand(-1, -1, self.dim)  # [B, L, dim]

        # 执行替换
        trans = torch.where(mask, hate_row, trans)
        # trans[i == self.dim - 1] = (1 - self.gamma) * one_minus_alpha
        # scatter 加入 α 到对应位置
        trans = trans.scatter_add(-1, i.unsqueeze(-1), alpha.unsqueeze(-1))
        return trans  # [B, 1, dim]
        # B = i.shape[0]
        # device = i.device

        # alpha = (-int_beta).exp()  
        # identity = F.one_hot(i, num_classes=self.dim).float()  

        # base = torch.full((B, self.dim), self.gamma / (self.dim - 1), device=device)
        # base[i == self.dim - 1] = 1 - self.gamma  # 对每一行检查是否 i == dim - 1

        # edge = alpha.unsqueeze(-1) * identity + (1 - alpha).unsqueeze(-1) * base
        # return edge 

    def sample_prob(self, i, int_beta):
        # 移动的总概率（来自 int_beta）
        move_chance = 1 - (-int_beta).exp()
        move_mask = torch.rand_like(int_beta, dtype=torch.float, device=i.device) < move_chance

        # 针对“决定移动”的样本，使用 self.gamma 决定是去往 [0, dim-2] 还是 dim-1
        sample_gamma = torch.rand_like(i, dtype=torch.float, device=i.device)

        # 随机采样 [0, dim - 2] 的位置（当选择非最后一类时使用）
        non_final = torch.randint(0, self.dim - 1, i.shape, device=i.device)

        # 默认不移动
        i_pert = i.clone()

        # 根据 gamma 决定移动位置
        move_pairwise = (sample_gamma < self.gamma) & move_mask
        move_pointwise = (~move_pairwise) & move_mask

        i_pert[move_pairwise] = non_final[move_pairwise]
        i_pert[move_pointwise] = self.dim - 1

        return i_pert
    
    @torch.no_grad()
    def reverse_prob_ratio(self, exp_score, minus_beta):
        """
        Computes the approximated ratios of P_{s} / P_{t}(x_t=i) as P_{t|s}^{-1} * exp_Rankings(x_t=i, t)
        P_{t|s}^{-1} = inverse_alpha * I + (1 - inverse_alpha) * E 
        """
        inverse_alpha = minus_beta.exp()
        #score = score.clone() # yeah yeah whatever we should probably do this
        extra_const1 = (1 - inverse_alpha) * (1 - self.gamma) * exp_score.sum(dim=-1)
        extra_const2 = (1 - inverse_alpha) * self.gamma / self._dim * exp_score.sum(dim=-1)
        exp_score *= inverse_alpha.unsqueeze(-1)
        exp_score[..., :-1] += extra_const2
        exp_score[..., -1] += extra_const1
        return exp_score 

    def reverse_prob_ratio(self, score, beta):
        dim = score.shape[-1]
        epow = (-beta).exp()[..., None]

        E_row = torch.full_like(score, self.gamma / self._dim)
        E_row[..., -1] = 1 - self.gamma
        sum_score = (E_row * score).sum(dim=-1, keepdim=True)

        return ((epow - 1) * sum_score / dim) + score / epow

    def sample_nonpreference(self, *batch_dims):
        return torch.full(batch_dims, self.dim - 1, dtype=torch.int64)

    def sample_nonpreference(self, *batch_dims):
        # 随机采样，决定是否取最后一类（dim - 1）
        sample_mask = torch.rand(*batch_dims) < self.gamma
        # 在 [0, dim - 2] 范围内随机采样
        non_final = torch.randint(0, self.dim - 1, batch_dims)
        # 最终结果：gamma 概率为 dim - 1，剩下随机取其他类
        return torch.where(sample_mask, torch.full(batch_dims, self.dim - 1), non_final)
    
    def score_entropy(self, score, int_beta, x, x0):
        # print("x0:",x0)
        # print("xt:",x)
        # print("int_beta:",int_beta)
        # print("score:",score)
        x0 = x0.unsqueeze(-1)
        #score = score.squeeze()
        # esigm1 = (1-alpha_t) / alpha_t
        esigm1 = torch.where(
            int_beta < 0.5,
            torch.expm1(int_beta),
            torch.exp(int_beta) - 1
        )
        ratio_base = (1 - self.dim / ( self.gamma * esigm1 + self.dim))

        score_zero = torch.gather(score, -1, x[..., None]).squeeze(-1)
        #print("score zero:", score_zero)

        score_target = torch.gather(score, -1, x0[..., None]).squeeze(-1)

        neg_term_base = (score.sum(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1)) / (self.dim - 1)
        # neg_term_like = self.gamma * neg_term_base + (1 - self.gamma * (1 + 1. / (self.dim - 1))) * score[...,-1]
        # neg_term_mild = self.gamma * neg_term_base + ( 1. / ratio_base - 1) * self.gamma / (self.dim - 1) * torch.gather(score, -1, x0[..., None]) + \
        #                 (1 - self.gamma  * (1 + 1. / (self.dim - 1))) * score[...,-1]
        # neg_term_hate = self.gamma * neg_term_base + ( 1. / ratio_base - 1) * self.gamma / (self.dim - 1) * torch.gather(score, -1, x0[..., None])
        # print(score.shape)
        # print(x0.shape)
        # print(x.shape)
        # print(esigm1.shape)
        # print(neg_term.shape)
        #neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim

        # neg_term = torch.where(
        #     x == x0,
        #     ratio_base * neg_term_like,
        #     torch.where(
        #         x == self.dim - 1,
        #         neg_term_hate,
        #         neg_term_mild
        #     )
        # )

        # neg_term = torch.where(
        #     x == x0,
        #     ratio_base * (self.gamma * neg_term_base + (1 - self.gamma * (1 + 1. / (self.dim - 1))) * score[...,-1]),
        #     torch.where(
        #         x == self.dim - 1,
        #         self.gamma * neg_term_base + ( 1. / ratio_base - 1) * self.gamma / (self.dim - 1) * torch.gather(score, -1, x0[..., None]),
        #         self.gamma * neg_term_base + ( 1. / ratio_base - 1) * self.gamma / (self.dim - 1) * torch.gather(score, -1, x0[..., None]) + \
        #                  (1 - self.gamma  * (1 + 1. / (self.dim - 1))) * score[...,-1]
        #     )
        # )
        # like 情况 (x == x0)
        neg_like = ratio_base * (
            self.gamma * neg_term_base +
            (1 - self.gamma * (1 + 1. / (self.dim - 1))) * score[..., -1]
        )
        # hate 情况 (x == dim - 1)
        neg_hate = self.gamma * neg_term_base + \
            (1. / ratio_base - 1) * self.gamma / (self.dim - 1) * score_target
        # mild 情况 (其他)
        neg_mild = self.gamma * neg_term_base + \
            (1. / ratio_base - 1) * self.gamma / (self.dim - 1) * score_target + \
            (1 - self.gamma * (1 + 1. / (self.dim - 1))) * score[..., -1]

        # 合并逻辑判断（矢量化）
        neg_term = torch.where(
            x == x0,
            neg_like,
            torch.where(
                x == self.dim - 1,
                neg_hate,
                neg_mild
            )
        )

        # print("neg_term:", neg_term)

        gamma_const = (self.gamma) / ((1 - self.gamma) * (self.dim - 1))

        # constant factor
        const = torch.where(
            x == x0,
            (1 - self.gamma / (self.dim - 1)) * ratio_base * (ratio_base.log() - 1.) - (1 - self.gamma) * ratio_base * gamma_const.log(),
            torch.where(
                x == self.dim - 1,
                self.gamma * (1 + (1. / ratio_base - 1.) / (self.dim - 1)) * (gamma_const.log() - 1.) - ratio_base.log() / ratio_base,
                -(1 - self.gamma) * (gamma_const.log() + 1.) - (ratio_base.log() + self.gamma.log() + 1.) / (ratio_base * (self.dim - 1)) - self.gamma * (1 - 2. / (self.dim - 1))
            )
        )

        # print("const:", const)

        #positive term
        sexp = score.exp()
        pos_term_base = (sexp.sum(dim=-1) - 1) / (self.dim - 1)
        #pos_term_base = sexp.mean(dim=-1) * ( 1 + 1. / (self.dim - 1)) - 1. / (self.dim - 1)
        pos_term = torch.where(
            x == self.dim - 1,
            (self.dim - 1) * (1 - self.gamma) * pos_term_base,
            self.gamma * pos_term_base
        )
        # print("pos_term", pos_term)
        return pos_term - neg_term + const

class AdaptiveWise(PreferGrow, nn.Module):
    def __init__(self, dim, is_disliked_item = True):
        PreferGrow.__init__(self)
        nn.Module.__init__(self)
        self._dim = dim
        self._is_disliked_item = is_disliked_item
        self.init_adaptive_probs()

    @property
    def is_disliked_item(self):
        return self._is_disliked_item
    @property
    def dim(self):
        return self._dim + (1 if self.is_disliked_item else 0)
    
    def init_adaptive_probs(self):
        self.p1 = nn.Parameter(torch.ones(self.dim))
    
    def nonpreference_probs(self,):
        return F.softmax(self.p1, dim=-1)
    
    def rate_matrix_col(self, i):
        hate_probs = self.nonpreference_probs().to(i.device)
        hate_probs = hate_probs.unsqueeze(0).unsqueeze(0).expand(*i.shape, -1)
        rate = hate_probs.scatter_add(-1, i.unsqueeze(-1), torch.full_like(i.unsqueeze(-1), -1))
        return rate

    def rate_matrix_row(self, i):
        hate_probs = self.nonpreference_probs().to(i.device)
        hate_probs_row = hate_probs[i].unsqueeze(-1).expand(-1, -1, self.dim)
        rate = hate_probs_row.scatter_add(-1, i.unsqueeze(-1), torch.full_like(i.unsqueeze(-1), -1))
        return rate

    def prob_matrix_col(self, i, int_beta):
        hate_probs = self.nonpreference_probs().to(i.device)
        hate_probs = hate_probs.unsqueeze(0).unsqueeze(0)
        alpha = (-int_beta[..., None]).exp()
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - alpha) * hate_probs
        trans = trans.scatter_add(-1, i[..., None], alpha)
        return trans
    
    def prob_matrix_row(self, i, int_beta):
        hate_probs = self.nonpreference_probs().to(i.device)
        alpha = (-int_beta[..., None]).exp()
        trans = hate_probs[i].unsqueeze(-1).expand(-1, -1, self.dim) * (1 - alpha)
        trans = trans.scatter_add(-1, i[..., None], alpha)
        return trans

    def sample_prob(self, i, int_beta):
        move_chance = 1. - (-int_beta).exp()
        # Move chance ratio (n-1)/n * (1. - (-sigma).exp())
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        probs = self.nonpreference_probs()  # [dim]
        B, L = i.shape
        samples = torch.multinomial(probs, B * L, replacement=True).view(B, L).to(i.device)  # [B * L]
        i_pert = torch.where(move_indices, samples, i)  # [B, L]
        return i_pert

    @torch.no_grad()
    def reverse_prob_ratio(self, exp_score, minus_beta):
        """
        Computes the approximated ratios of P_{s} / P_{t}(x_t=i) as P_{t|s}^{-1} * exp_Rankings(x_t=i, t)
        P_{t|s}^{-1} = inverse_alpha * I + (1 - inverse_alpha) * E 
        """
        #print(minus_beta.shape)
        inverse_alpha = minus_beta.exp()
        # print(inverse_alpha.shape)
        hate_probs = self.nonpreference_probs().to(exp_score.device)
        #
        #score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - inverse_alpha) * exp_score.sum(dim=-1)
        #print(exp_score.shape)
        #print(hate_probs.shape)
        return inverse_alpha.unsqueeze(-1) * exp_score + extra_const.unsqueeze(-1) * hate_probs.unsqueeze(0).unsqueeze(0).expand(*exp_score.shape)

    def sample_nonpreference(self, *batch_dims):
        probs = self.nonpreference_probs()  # [dim]
        B, L = batch_dims
        return      torch.multinomial(probs, B * L, replacement=True).view(B, L)

    def score_entropy(self, score, int_beta, x, x0):
        hate_probs = self.nonpreference_probs().to(score.device)  # [dim]
        x0 = x0.unsqueeze(-1)
        #score = score.squeeze()
        # esigm1 = (1-alpha_t) / alpha_t
        esigm1 = torch.where(
            int_beta < 0.5,
            torch.expm1(int_beta),
            torch.exp(int_beta) - 1
        )
        ratio_base0 = 1. / esigm1
        ratio_base1 = esigm1 * hate_probs[x]
        ratio_base2 = (1 - 1. / (1 + ratio_base1)) 

        # negative term
        # note that s_theta(x_t,t,u)_{x_t} == 0 which already done in model output
        neg_term_base = (score * (hate_probs.unsqueeze(0).unsqueeze(0))).sum(dim=-1)

        neg_term = torch.where(
            x == x0,
            ratio_base2 * neg_term_base,
            neg_term_base + torch.gather(score, -1, x0.unsqueeze(-1)).squeeze(-1) * ratio_base0 
        )

        #print("neg_term.shape =", neg_term.shape)

        # constant factor
        const_base = (hate_probs * hate_probs.log()).sum(dim=-1)
        const = torch.where(
            x == x0,
            ratio_base2 * ( const_base + hate_probs[x] * hate_probs[x].log() + (hate_probs[x] - 1) * ((ratio_base1 + 1.).log() + ratio_base0.log() - 1)),
            const_base + hate_probs[x] + (hate_probs[x0] + ratio_base0) * ((esigm1 * hate_probs[x0] + 1.).log() + ratio_base0.log()) - \
                (1 + ratio_base0) *(hate_probs[x].log() + 1.) 
        )

        #positive term
        sexp = score.exp()
        #  torch.gather(sexp, -1, x[..., None]).squeeze(-1)
        pos_term = (sexp.sum(dim=-1) - 1.) * hate_probs[x]
        return (pos_term - neg_term + const)
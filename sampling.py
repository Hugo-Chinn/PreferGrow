import abc
import torch
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass

@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, h, x, t, step_size, denoise=False):
        sigma, dsigma = self.noise(t)
        score = score_fn(h, x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        #x = self.graph.sample_rate(x, rev_rate)
        if denoise:
            return F.one_hot(x, num_classes=self.dim).to(rev_rate) + rev_rate
        return sample_categorical(F.one_hot(x, num_classes=score.size(-1)).to(rev_rate) + rev_rate)

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, h, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, h, x, t, step_size, denoise=False, sample_method="hard"):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(h, x, curr_sigma)

        #print("Ranking_t", score)

        stag_score = self.graph.reverse_prob_ratio(score, dsigma)
        probs = stag_score * self.graph.prob_matrix_row(x, dsigma)
        #print("Reverse probs", probs)
        if denoise:
            return probs
        return sample_categorical(probs, sample_method)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, h, x, t, denoise=False):
        sigma = self.noise(t)[0]

        score = score_fn(h, x, sigma)
        #print(score.size())
        #print(score.size())
        stag_score = self.graph.reverse_prob_ratio(score, sigma)
        #print(stag_score.size())
        probs = stag_score * self.graph.prob_matrix_row(x, sigma)
        #print(probs.size())
        # truncate probabilities
        if self.graph.is_disliked_item:
            probs = probs[..., :-1]
        if denoise:
            return probs
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, eps, personalization_strength, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 personalization_strength=personalization_strength,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn

def get_eval_score(config, graph, noise, eps, device):
    
    eval_score_fn = get_one_step_score(graph=graph,
                                     noise=noise,
                                     denoise=config.sampling.noise_removal,
                                     device=device)
    
    return eval_score_fn
    
def get_one_step_score(graph, noise, denoise=True, device=torch.device('cpu'), proj_fun=lambda x: x):
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def one_step_score(model, history, x_0, t, mc_sampling_number):
        """
        Args:
            model: predictor model
            batch_dims: ignored here, or can be used for reshaping
            history: user history or context
            x_0: ground truth input (shape = [B] or [B, d])
            t: timestep, usually scalar or tensor [B]
            mc_sampling_number: number of MC samples
        Returns:
            score_mean: [B]
            score_std : [B]
        """

        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        sigma, dsigma = noise(t)   # [B]

        score_of_x_0 = []

        for i in range(mc_sampling_number):
            # Sample x_t
            if len(x_0.size()) == 1:
                x_t = graph.sample_transition(x_0[:, None], sigma[:, None])   # [B, d]
            elif len(x_0.size()) == 2:
                x_t = graph.sample_transition(x_0, sigma[:, None])            # [B, d]

            x_t = projector(x_t)

            # Compute denoised logits
            logits = denoiser.update_fn(
                sampling_score_fn,
                history,
                x_t,
                t * torch.ones(x_t.shape[0], 1, device=device),
                denoise=denoise
            )   # shape [B, vocab_size] or similar

            # collect score logits corresponding to x_0 position
            # gathered = logits[x_0].squeeze()   # [B]
            logits_normalized = torch.softmax(logits, dim=-1)
            gathered = logits_normalized.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)
            score_of_x_0.append(gathered)

        # shape [mc_sampling_number, B]
        score_of_x_0 = torch.stack(score_of_x_0, dim=0)

        # compute mean & std along mc dimension
        score_mean = score_of_x_0.mean(dim=0)  # [B]
        score_std  = score_of_x_0.std (dim=0)  # [B]

        return score_mean, score_std

    return one_step_score


# def get_pc_sampler(graph, noise, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
#     predictor = get_predictor(predictor)(graph, noise)
#     projector = proj_fun
#     denoiser = Denoiser(graph, noise)

#     @torch.no_grad()
#     def pc_sampler(model, batch_dims, history):
#         sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
#         x = graph.sample_limit(*batch_dims).to(device)
#         timesteps = torch.linspace(1, eps, steps + 1, device=device)
#         dt = (1 - eps) / steps

#         for i in range(steps):
#             t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
#             x = projector(x)
#             x = predictor.update_fn(sampling_score_fn, history, x, t, dt)
            

#         if denoise:
#             # denoising step
#             x = projector(x)
#             t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
#             logits = denoiser.update_fn(sampling_score_fn, history, x, t, denoise=True)
            
#         return logits
    
#     return pc_sampler

def get_pc_sampler(graph, noise, predictor, steps, personalization_strength, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model, batch_dims, history):
        sampling_score_fn = mutils.get_score_fn(model, personalization_strength, train=False, sampling=True)
        x = graph.sample_nonpreference(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            #print("Timestep", timesteps[i])
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, history, x, t, dt, denoise=False, sample_method="hard")

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            logits = denoiser.update_fn(sampling_score_fn, history, x, t, denoise=True)
            
        return logits
    
    return pc_sampler

def get_full_sampler(graph, noise, predictor, steps, personalization_strength, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model, batch_dims, history):
        sampling_score_fn = mutils.get_score_fn(model, personalization_strength, train=False, sampling=True)
        x = graph.sample_nonpreference(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        probs_full = []
        for i in range(steps):
            #print("Timestep", timesteps[i])
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            probs = predictor.update_fn(sampling_score_fn, history, x, t, dt, denoise=True)
            probs_full.append(probs.squeeze(1))
            x = sample_categorical(probs, method = 'hard')

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            logits = denoiser.update_fn(sampling_score_fn, history, x, t, denoise=True)
        
        probs_full.append(logits.squeeze(1))
        # import pickle
        # with open('draw_data/full_sample_probs_list.pkl', 'wb') as f:
        #     pickle.dump(probs_full, f)

        return probs_full
    
    return pc_sampler

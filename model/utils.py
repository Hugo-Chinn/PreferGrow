import torch
import torch.nn.functional as F


def get_model_fn(model, personalized_strength, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability 

    Returns:
        A model function.
    """

    def model_fn(h, x, sigma):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()
            return model.forward_eval(h, x, sigma, personalized_strength )

        
            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(h, x, sigma)

    return model_fn


def get_score_fn(model, personalized_strength, train=False, sampling=False):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, personalized_strength, train=train)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def score_fn(h, x, sigma):
            sigma = sigma.reshape(-1)
            score = model_fn(h, x, sigma)
            # score = model_fn(h, x, sigma)
            
            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp()
                
            return score

    return score_fn

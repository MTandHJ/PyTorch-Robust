



import torch.nn.functional as  F
import eagerpy as ep
from foolbox.attacks import LinfProjectedGradientDescentAttack


class LinfPGDKLDiv(LinfProjectedGradientDescentAttack):

    # kl divergence as the loss function ...
    def get_loss_fn(self, model, logits_p):
        def loss_fn(inputs):
            logits_q = model(inputs)
            return ep.kl_div_with_logits(logits_p, logits_q).sum()
        return loss_fn

class LinfPGDSoftmax(LinfProjectedGradientDescentAttack):

    # the model returns the probs after softmax ...
    def get_loss_fn(self, model, labels):
        def loss_fn(inputs):
            probs = model(inputs)
            loss = F.nll_loss(probs.log().raw, labels.raw)
            return ep.astensor(loss)
        return loss_fn


















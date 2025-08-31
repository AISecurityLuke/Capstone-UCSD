import tensorflow as tf
import torch
import torch.nn.functional as F
from tensorflow.keras.utils import register_keras_serializable

# ---------------------------------------------------------------------------
# Keras serialisable focal-loss implementation
# ---------------------------------------------------------------------------

@register_keras_serializable(package="Custom")
class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Categorical focal-loss that *round-trips* through Keras’ save/load stack.

    Parameters
    ----------
    gamma : float, default 2.0
        Focusing parameter that down-weights easy examples.
    alpha : list[float] | None, default None
        Per-class weighting factor.  Length must equal ``num_classes``.
    reduction : str | tf.keras.losses.Reduction, default "sum_over_batch_size"
        Standard Keras reduction argument.  MUST be accepted so that the
        deserializer can construct the object from its config dict.
    name : str, default "categorical_focal_loss"
        Name for the loss tensor in Keras graphs.
    """

    def __init__(self, gamma: float = 2.0, alpha=None,
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name: str = "categorical_focal_loss", **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.gamma = gamma
        self.reduction_type = reduction  # store for get_config
        if alpha is not None:
            self.alpha = tf.constant(alpha, dtype=tf.float32)
        else:
            self.alpha = None

    def call(self, y_true, y_pred):  # pylint: disable=arguments-differ
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred)
        if self.alpha is not None:
            cross_entropy = cross_entropy * self.alpha

        weight = tf.pow(1.0 - y_pred, self.gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "gamma": self.gamma,
            "alpha": (self.alpha.numpy().tolist() if self.alpha is not None else None),
            "reduction": self.reduction_type,
        })
        return cfg

# ---------------------------------------------------------------------------
# Factory helper (backwards-compatible with previous signature)
# ---------------------------------------------------------------------------

def get_keras_focal_loss(gamma: float = 2.0, alpha=None):
    """Return a **serialisable** focal-loss instance for Keras models."""
    return CategoricalFocalLoss(gamma=gamma, alpha=alpha)


class TorchFocalLoss(torch.nn.Module):
    """Focal loss for multi-class classification (PyTorch).

    Args:
        gamma (float): focusing parameter.
        alpha (torch.Tensor or None): per-class weights (shape [C]). Use same device as logits.
        reduction (str): 'mean' or 'sum'.
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # Ensure alpha is on the same device as logits
        weight = None
        if self.alpha is not None:
            weight = self.alpha.to(logits.device)

        # Apple-Silicon MPS backend currently does **not** support the `weight` argument
        # in torch.nn.functional.cross_entropy.  Detect that case and fall back to a
        # manual class-weight scaling so we can still run on MPS without crashing.
        try:
            ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction='none')
        except RuntimeError as e:
            if 'Placeholder storage has not been allocated on MPS' in str(e):
                # Compute unweighted CE and scale manually
                ce_loss = F.cross_entropy(logits, targets, reduction='none')
                if weight is not None:
                    ce_loss = ce_loss * weight[targets]
            else:
                raise
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss.mean() 
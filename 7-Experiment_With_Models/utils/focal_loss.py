import tensorflow as tf
import torch
import torch.nn.functional as F


def get_keras_focal_loss(gamma: float = 2.0, alpha=None):
    """Return a Keras-compatible categorical focal loss function.

    Args:
        gamma (float): focusing parameter.
        alpha (list or None): per-class weight list; length must equal num_classes.
    """

    alpha_tensor = None
    if alpha is not None:
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def focal_loss(y_true, y_pred):
        # adaptive_trainer passes one-hot vectors (shape: [batch, num_classes]).
        # Cast to float32 to be safe.
        y_true_one_hot = tf.cast(y_true, tf.float32)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)

        if alpha_tensor is not None:
            cross_entropy = cross_entropy * alpha_tensor

        weight = tf.pow(1.0 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    return focal_loss


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
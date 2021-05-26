import torch


def CEL_label_smoothing(pred, target, smoothing=0., mask=None):
    batch, n_classes = pred.shape

    pred = pred.log_softmax(dim=-1)
    oh_mask = torch.arange(n_classes).unsqueeze(0).expand(batch, n_classes).to(pred.device)
    exp_target = target.unsqueeze(-1).expand(batch, n_classes)
    one_hot = exp_target.eq(oh_mask)

    base = torch.ones(batch, n_classes).to(pred.device) * smoothing / (n_classes - 1)
    smoothed = base + one_hot * (1 - smoothing - smoothing / (n_classes - 1))

    loss = -pred * smoothed
    loss = loss.sum(-1)

    if mask is not None:
        loss = loss * mask
        return loss.sum()/mask.sum()
    
    return loss.mean()

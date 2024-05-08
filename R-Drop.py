import torch
import torch.nn as nn
import torch.nn.functional as F


class RDrop(nn.Module):
    def __init__(self, reduction='mean') -> None:
        """
        reduction (str, optional): Specifies the reduction to apply to the loss: 
            - mean | sum | none
        """
        super().__init__()
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.kl_div = nn.KLDivLoss(reduction='none')

    def forward(self, logits1, logits2, target, kl_weight=1.0):
        ce_loss = (self.ce_loss(logits1, target) + self.ce_loss(logits2, target))/2
        kl_div_1 = self.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_div_2 = self.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_div_1+kl_div_2)/2
        
        loss = ce_loss + kl_weight*kl_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    logits1 = torch.randn(32, 2)
    logits2 = torch.randn(32, 2)
    target = torch.randint(0, 2, size=(32,))

    criterion = RDrop(reduction="mean")
    print(criterion(logits1, logits2, target))

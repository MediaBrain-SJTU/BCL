import torch
import torch.nn.functional as F
import torch.nn as nn

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def nt_xent(x, t=0.5, features2=None, average=True):

    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    # estimator g()
    Ng = neg.sum(dim=-1)
    
    # contrastive loss
    loss = (- torch.log(pos / (pos + Ng)))
    loss_reshape = loss.view(2, batch_size).mean(0)

    if average:
        return loss.mean()
    else: 
        return loss_reshape

class NT_Xent_Loss(nn.Module):
    def __init__(self, temp=0.2, average=True):
        super(NT_Xent_Loss, self).__init__()
        self.temp = temp
        self.average = average

    def forward(self, features, features_=None):
        return nt_xent(features, t=self.temp, features2=features_, average=self.average)
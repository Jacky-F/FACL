import torch
import torch.nn as nn
from torch.nn import parameter
from models import FACL
import torch.nn.functional as F
import numpy as np
import os
import ipdb
import scipy.stats


def initModel(mod, gpu_ids):
    mod = mod.to(f'cuda:{gpu_ids[0]}')
    mod = nn.DataParallel(mod, gpu_ids)
    return mod


class DefakeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(DefakeContrastiveLoss, self).__init__()
        self.temperature = torch.tensor(temperature)
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask).cuda()
        return (M * mask).view(h, -1)

    def forward(self, x, label):
        # x: (n fea)
        # label: (n) real 0 fake 1

        re_label = 1 - label
        logits = self.calculate_similarity_matrix(x, x)

        real_label_mask = re_label[:, None] @ re_label[None]
        fake_label_mask = (1 - torch.eq(label[:, None], label[None]).float())

        all_sim = torch.exp(self._remove_diag(logits) / self.temperature)
        real_sim = (all_sim * real_label_mask).sum(dim=1)
        fake_sim = (all_sim * fake_label_mask).sum(dim=1)

        frac = real_sim / (all_sim.sum(dim=1) + fake_sim)
        frac = frac[frac > 0]
        loss = -torch.log(frac).mean()
        return loss


class Trainer(): 
    def __init__(self, gpu_ids, mode, w1, w2, lr):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.model = FACL(mode=mode)
        self.model = initModel(self.model, gpu_ids)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.mode = mode
        self.loss_mask_fn = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.defakeContrastive = DefakeContrastiveLoss(temperature=0.07)
        self.w1 = w1
        self.w2 = w2

    def set_input(self, input, label, mask):
        self.input = input.to(self.device)
        self.label = label.to(self.device)
        self.mask = mask.to(self.device)
        self.defakeContrastive = self.defakeContrastive.to(self.device)

    def forward(self, x):
        fea, out, pred_mask = self.model(x)
        return out
    
    def optimize_weight(self):
        stu_fea, stu_cla, pred_mask = self.model(self.input)

        self.loss_cla = self.loss_fn(stu_cla.squeeze(1), self.label)  # classify loss
        if pred_mask is not None:
            self.loss_mask = self.loss_mask_fn(pred_mask+1e-10, self.mask)
        else:
            self.loss_mask = 0
        self.metric_loss = self.defakeContrastive(stu_fea, self.label)
        if self.mode == 'Two_Stream':
            self.loss = self.loss_mask * self.w1 + self.loss_cla + self.metric_loss * self.w2
        elif self.mode == 'FAM' or self.mode == 'Original':
            self.loss = self.loss_cla
        elif self.mode == 'Decoder':
            self.loss = self.loss_cla + self.loss_mask * self.w1
        else:
            # mode == 'CLloss'
            self.loss = self.loss_cla + self.metric_loss * self.w2

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict, strict=False)



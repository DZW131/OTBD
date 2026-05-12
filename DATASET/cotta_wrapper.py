import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CoTTAWrapper(nn.Module):
    def __init__(self, model: nn.Module, restore_p: float = 0.05, ema_m: float = 0.95):
        super().__init__()
        # student and teacher models
        self.student = model
        self.teacher = copy.deepcopy(model)
        self.restore_p = restore_p
        self.ema_m = ema_m
        self._adapt_dtype = None  # track dtype alignment
        self.ema_decay = 0.99

        self.activated = False
        
        # freeze teacher params, ensure student trainable
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.student.parameters():
            p.requires_grad = True

    def set_activated_mod(self, check: bool):
        self.activated = check

    @torch.no_grad()
    def ema_update(self):
        # EMA update of teacher parameters
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(self.ema_m).add_(s_param.data * (1 - self.ema_m))

    def stochastic_restore(self):
        # randomly restore some student weights to original teacher init
        orig = dict(self.teacher.named_parameters())
        for name, param in self.student.named_parameters():
            if random.random() < self.restore_p:
                param.data.copy_(orig[name].data)

    @torch.no_grad()
    def update_ema(self, student_model, teacher_model, alpha=0.95):
        for s_param, t_param in zip(student_model.parameters(), teacher_model.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data * (1 - alpha))


    @torch.enable_grad()
    def forward(self, x):
        return self.student(x)

    def get_unlabeled_student_transform(self):
        return Compose3D([
            AddGaussianNoise3D(std=0.05, prob=0.5),
            RandomContrast3D(gamma_range=(0.7, 1.5), prob=0.5)
        ])

    def get_unlabeled_teacher_transform(self):
        return Compose3D([
            AddGaussianNoise3D(std=0.05, prob=0.2)
        ])

    @torch.enable_grad()
    def adapt_batch(self, imgs: torch.Tensor):
        """
        One CoTTA adaptation step. Runs with grad enabled even inside no_grad contexts.
        Args:
            imgs: input batch tensor
            device: torch.device
        Returns:
            loss value
        """
        self.student.train()
        # imgs = imgs.to(device)
        device = imgs.device
        # align model dtype with input dtype once
        if self._adapt_dtype is None or imgs.dtype != self._adapt_dtype:
            self._adapt_dtype = imgs.dtype
            # cast student and teacher to input dtype
            self.student = self.student.to(device=device, dtype=self._adapt_dtype)
            # The teacher is frozen but needs to match dtype
            self.teacher = self.teacher.to(device=device, dtype=self._adapt_dtype)

        img_u_s = []
        img_u_t = []
        for i in range(imgs.shape[0]):
            # img_u_s.append(self.get_unlabeled_student_transform()({'image': imgs[i]})["image"])
            # img_u_t.append(self.get_unlabeled_teacher_transform()({'image': imgs[i]})["image"])
            transform = self.get_unlabeled_student_transform()
            img_u_s.append(transform(imgs[i]))
            transform = self.get_unlabeled_teacher_transform()
            img_u_t.append(transform(imgs[i]))
        
        img_u_s = torch.stack(img_u_s,dim=0)
        img_u_t = torch.stack(img_u_t,dim=0)
        # generate pseudo-labels with teacher in no_grad
        with torch.no_grad():
            preds = []
            for _ in range(4):
                xi = img_u_t
                # if random.random() < 0.5:
                #     xi = imgs + torch.randn_like(imgs) * 0.01
                pi = self.teacher(xi)
                preds.append(F.softmax(pi, dim=1))
            pseudo_label = torch.stack(preds).mean(0).argmax(1)

        # student update
        logits = self.student(img_u_s)
        # check requires_grad on logits
        # print(f"[CoTTA] pseudo_label.requires_grad={pseudo_label.requires_grad}")
        # print(f"[CoTTA] logits.requires_grad={logits.requires_grad}")
        # print("全局 grad 开关:", torch.is_grad_enabled())
        # print("imgs.requires_grad:", imgs.requires_grad)
        # print("student parameters require_grad:", any(p.requires_grad for p in self.student.parameters()))
        loss = F.cross_entropy(logits, pseudo_label)
        loss.backward()

        # restore and update
        self.stochastic_restore()
        self.update_ema(self.student, self.teacher, alpha=self.ema_decay)

        return loss.item()



class AddGaussianNoise3D:
    def __init__(self, std=0.05, prob=0.5):
        self.std = std
        self.prob = prob

    def __call__(self, x: torch.Tensor):
        # x: Tensor [C, D, H, W]
        if random.random() < self.prob:
            noise = torch.randn_like(x, device=x.device) * self.std
            x = x + noise
            x = torch.clamp(x, 0.0, 1.0)
        return x

class RandomContrast3D:
    def __init__(self, gamma_range=(0.7, 1.5), prob=0.5):
        self.gamma_range = gamma_range
        self.prob = prob

    def __call__(self, x: torch.Tensor):
        if random.random() < self.prob:
            gamma = random.uniform(*self.gamma_range)
            # Clamp to avoid NaN when x has zeros
            x = torch.clamp(x, 1e-6, 1.0)
            x = x ** gamma
            x = torch.clamp(x, 0.0, 1.0)
        return x

class Compose3D:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor):
        for t in self.transforms:
            x = t(x)
        return x
from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import json
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule, CosineSchedulerIter
from torch.autograd import Variable, Function


class Prompt(NormalNN):
    def __init__(self, learner_config):
        self.prompt_param = learner_config["prompt_param"]
        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets):

        # logits
        logits, prompt_loss = self.model(
            inputs, train=True, cls_mean=self.cls_mean
        )  # logits=cls_token if pen=True, else self.model.last(cls_token)
        logits = logits[:, : self.valid_out_dim]

        # ce with heuristic
        logits[:, : self.last_valid_out_dim] = -float(
            "inf"
        )  # TODO: this gives inf loss if self.memory_size > 0
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    # sets model optimizers
    def init_optimizer(self):

        if len(self.config["gpuid"]) > 1:
            base_params = list(self.model.module.prompt.parameters())
            base_fc_params = list(self.model.module.last.parameters())
        else:
            if self.model.prompt.task_count == 0:
                base_params = [
                    p
                    for name, p in self.model.prompt.named_parameters()
                    if p.requires_grad
                ]
            else:
                # base_params = [p for name, p in self.model.prompt.named_parameters() if 'mlp' not in name]
                # base_params = list(self.model.prompt.parameters())
                base_params = [
                    p
                    for name, p in self.model.prompt.named_parameters()
                    if p.requires_grad
                ]

            base_fc_params = list(self.model.last.parameters())
        base_params = {
            "params": base_params,
            "lr": self.config["lr"] * 5,
            "weight_decay": self.config["weight_decay"],
        }  # HiDe-Prompt - larger_prompt_lr
        base_fc_params = {
            "params": base_fc_params,
            "lr": self.config["lr"],
            "weight_decay": self.config["weight_decay"],
        }
        optimizer_arg = [base_params, base_fc_params]

        total_params = sum(p.numel() for p in self.model.parameters())
        print("Total params:", total_params)
        tune_params = sum(p.numel() for p in base_params["params"]) + sum(
            p.numel() for p in base_fc_params["params"]
        )
        print("Tune params:", tune_params)
        tuned_percent = tune_params / total_params * 100
        print(f"Tune ratio: {tuned_percent:.2f}")
        prompt_params = sum(p.numel() for p in base_params["params"])
        print("Prompt params:", prompt_params)
        prompt_percent = prompt_params / total_params * 100
        print(f"Prompt ratio: {prompt_percent:.2f}")

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config["optimizer"]](optimizer_arg)

        # create schedules
        if self.schedule_type == "cosine":
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == "decay":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.schedule, gamma=0.1
            )
        elif self.schedule_type == "coswm":
            # print(self.config)
            scheduler_cfg = {
                "base_value": [self.config["lr"] * 5, self.config["lr"]],
                "final_value": [1e-6, 1e-6],
                "optimizer": self.optimizer,
                "iter_step": self.config["iter_step"],
                "n_epochs": self.config["schedule"][-1],
                "last_epoch": -1,
                "warmup_epochs": self.config["schedule"][1],
                "start_warmup_value": 0,
                "freeze_iters": self.config["schedule"][0],
            }
            self.scheduler = CosineSchedulerIter(**scheduler_cfg)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config["gpuid"][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config["gpuid"]) > 1:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=self.config["gpuid"],
                output_device=self.config["gpuid"][0],
            )
        return self


# Our method
class VQPrompt(Prompt):
    def __init__(self, learner_config):
        super(VQPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="qt",
            prompt_param=self.prompt_param,
            pretrained=cfg["pretrained_weight"],
        )  # vit_pt_imnet
        return model


# Our method
class VAPTPrompt(Prompt):
    def __init__(self, learner_config):
        super(VAPTPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="vapt",
            prompt_param=self.prompt_param,
            pretrained=cfg["pretrained_weight"],
        )  # vit_pt_imnet
        return model

    def _score_log_path(self, task_id):
        log_dir = self.config.get("log_dir", "outputs")
        score_dir = os.path.join(log_dir, "expert_scores")
        os.makedirs(score_dir, exist_ok=True)
        return os.path.join(score_dir, f"expert_scores_task{task_id}.jsonl")

    def _log_scores(self, prompt_scores, task_id, batch_idx, epoch):
        if not hasattr(self, "score_log_file") or self.score_log_file is None:
            return

        lines = []
        e_layers = getattr(self.model.prompt, "e_layers", [])
        num_heads = getattr(self.model.prompt, "num_heads", 0)
        for e in e_layers:
            prompt_score, _ = prompt_scores[e]  # (B, num_heads, 1, num_experts)
            if prompt_score is None:
                continue
            avg_score = prompt_score.mean(dim=0).squeeze(1)  # (num_heads, num_experts)
            for h in range(min(num_heads, avg_score.size(0))):
                entry = {
                    "task_id": task_id,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "layer": e,
                    "head": h,
                    "scores": avg_score[h].detach().cpu().tolist(),
                }
                lines.append(json.dumps(entry))

        if len(lines) > 0:
            with open(self.score_log_file, "a") as f:
                f.write("\n".join(lines) + "\n")

    def init_optimizer(self, epoch_factor=1):

        if len(self.config["gpuid"]) > 1:
            base_params = list(self.model.module.prompt.parameters())
            base_fc_params = list(self.model.module.last.parameters())
        else:
            base_params = [
                p for name, p in self.model.prompt.named_parameters() if p.requires_grad
            ]

            base_fc_params = list(self.model.last.parameters())

        base_params = {
            "params": base_params,
            "lr": self.config["lr"] * 5,
            "weight_decay": self.config["weight_decay"],
        }  # HiDe-Prompt - larger_prompt_lr
        base_fc_params = {
            "params": base_fc_params,
            "lr": self.config["lr"],
            "weight_decay": self.config["weight_decay"],
        }
        optimizer_arg = [base_params, base_fc_params]

        total_params = sum(p.numel() for p in self.model.parameters())
        print("Total params:", total_params)
        tune_params = sum(p.numel() for p in base_params["params"]) + sum(
            p.numel() for p in base_fc_params["params"]
        )
        print("Tune params:", tune_params)
        tuned_percent = tune_params / total_params * 100
        print(f"Tune ratio: {tuned_percent:.2f}")
        prompt_params = sum(p.numel() for p in base_params["params"])
        print("Prompt params:", prompt_params)
        prompt_percent = prompt_params / total_params * 100
        print(f"Prompt ratio: {prompt_percent:.2f}")

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config["optimizer"]](optimizer_arg)
        num_epochs = int(self.config["schedule"][-1] * epoch_factor)

        # create schedules
        if self.schedule_type == "cosine":
            self.scheduler = CosineSchedule(self.optimizer, K=num_epochs)
        elif self.schedule_type == "decay":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.schedule, gamma=0.1
            )
        elif self.schedule_type == "coswm":
            # print(self.config)
            scheduler_cfg = {
                "base_value": [self.config["lr"] * 5, self.config["lr"]],
                "final_value": [1e-6, 1e-6],
                "optimizer": self.optimizer,
                "iter_step": self.config["iter_step"],
                "n_epochs": num_epochs,
                "last_epoch": -1,
                "warmup_epochs": self.config["schedule"][1],
                "start_warmup_value": 0,
                "freeze_iters": self.config["schedule"][0],
            }
            self.scheduler = CosineSchedulerIter(**scheduler_cfg)

    def update_model(self, inputs, targets, dense=False):
        # logits
        logits, prompt_loss = self.model(
            inputs, train=True, cls_mean=self.cls_mean, dense=dense
        )  # logits=cls_token if pen=True, else self.model.last(cls_token)

        logits = logits[:, : self.valid_out_dim]

        # ce with heuristic
        logits[:, : self.last_valid_out_dim] = -float(
            "inf"
        )  # TODO: this gives inf loss if self.memory_size > 0
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    def learn_prompt(self, train_loader, batch_time, dense=False, epoch_factor=1):

        losses = AverageMeter()
        acc = AverageMeter()

        batch_timer = Timer()
        num_epochs = int(self.config["schedule"][-1] * epoch_factor)

        if self.schedule_type == "coswm":  # step scheduler at each iter
            for epoch in range(num_epochs):
                self.epoch = epoch

                # for param_group in self.optimizer.param_groups:
                #     self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):
                    # verify in train mode
                    self.model.train()
                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    # model update
                    loss, output = self.update_model(x, y, dense=dense)
                    self.scheduler.step()

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(
                        output, y, task, acc, topk=(self.top_k,)
                    )  # already calculate train acc here? but logit range is narrow
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                # eval update
                self.log(
                    "Epoch:{epoch:.0f}/{total:.0f}".format(
                        epoch=self.epoch + 1, total=num_epochs
                    ),
                    end=" ",
                )
                self.log(
                    " * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}".format(
                        loss=losses, acc=acc
                    )
                )

                # reset
                losses = AverageMeter()
                acc = AverageMeter()
        else:
            for epoch in range(num_epochs):
                self.epoch = epoch

                if epoch > 0:
                    self.scheduler.step()
                # for param_group in self.optimizer.param_groups:
                #     self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    # model update
                    loss, output = self.update_model(x, y, dense=dense)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()

                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(
                        output, y, task, acc, topk=(self.top_k,)
                    )  # already calculate train acc here? but logit range is narrow
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                # eval update
                self.log(
                    "Epoch:{epoch:.0f}/{total:.0f}".format(
                        epoch=self.epoch + 1, total=num_epochs
                    )
                )
                self.log(
                    " * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}".format(
                        loss=losses, acc=acc
                    )
                )

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):

        task_id = self.task_count
        self.score_log_file = self._score_log_path(task_id)

        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
                # Cannot load, because in run.py, r<start_r is not allowed
                # all r in the loop, is not trained
                # I changed that in run.py to see effects
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log("Optimizer is reset!")
            self.init_optimizer()
        if need_train:

            if os.path.exists(self.score_log_file):
                os.remove(self.score_log_file)

            # data weighting
            self.data_weighting(train_dataset)

            batch_time = AverageMeter()

            if self.task_count == 0:
                print("-" * 10)
                print("Initial training...")
                self.log("Optimizer is reset!")
                epoch_factor = 0.5
                self.init_optimizer(epoch_factor=epoch_factor)
                self.learn_prompt(
                    train_loader, batch_time, dense=True, epoch_factor=epoch_factor
                )
                self.log("Optimizer is reset!")
                self.init_optimizer()

            self.learn_prompt(train_loader, batch_time)

            print("-" * 10)
            print("Selecting Experts...")
            num_samples = 0

            for i, (x, y, task) in enumerate(train_loader):
                # verify in train mode
                self.model.eval()
                # send data to gpu
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()

                with torch.no_grad():
                    prompt_scores = self.model(x, train=False, return_attn=True)

                self._log_scores(prompt_scores, task_id, batch_idx=i, epoch=-1)
                self.model.prompt.update_prompt(prompt_scores)
                num_samples += x.size(0)

            self.model.prompt.update_num_samples(num_samples)
            # self.model.prompt.print_freq()
            print("-" * 10)

        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(
                self.memory_size, np.arange(self.last_valid_out_dim)
            )

        try:
            return batch_time.avg, need_train
        except:
            return None, need_train


class SoftMoEPrompt(VAPTPrompt):
    def __init__(self, learner_config):
        super(SoftMoEPrompt, self).__init__(learner_config)
        self.loss_log_file = None

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="soft-moe",
            prompt_param=self.prompt_param,
            pretrained=cfg["pretrained_weight"],
        )
        return model

    def _prompt_module(self):
        if len(self.config["gpuid"]) > 1:
            return self.model.module.prompt
        return self.model.prompt

    def _feat_module(self):
        if len(self.config["gpuid"]) > 1:
            return self.model.module.feat
        return self.model.feat

    def _score_log_path(self, task_id):
        log_dir = self.config.get("log_dir", "outputs")
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"softmoe_scores_task{task_id}.jsonl")

    def _loss_log_path(self, task_id):
        log_dir = self.config.get("log_dir", "outputs")
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"softmoe_losses_task{task_id}.jsonl")

    def _log_batch_scores(self, prompt_scores, task_id, epoch, phase, phase_epoch, batch_idx):
        if not hasattr(self, "score_log_file") or self.score_log_file is None:
            return

        if prompt_scores is None:
            return

        lines = []
        prompt_module = self._prompt_module()
        e_layers = getattr(prompt_module, "e_layers", [])
        num_heads = getattr(prompt_module, "num_heads", 0)
        for e in e_layers:
            prompt_score, prompt_sigma = prompt_scores[e]
            if prompt_score is None:
                continue

            batch_score = prompt_score.mean(dim=0).squeeze(1)
            batch_sigma = (
                prompt_sigma.mean(dim=0).squeeze(1)
                if prompt_sigma is not None
                else None
            )
            for h in range(min(num_heads, batch_score.size(0))):
                entry = {
                    "task_id": task_id,
                    "phase": phase,
                    "epoch": epoch,
                    "phase_epoch": phase_epoch,
                    "batch": batch_idx,
                    "layer": e,
                    "head": h,
                    "scores": batch_score[h].detach().cpu().tolist(),
                }
                if batch_sigma is not None:
                    entry["sigmoid"] = batch_sigma[h].detach().cpu().tolist()
                lines.append(json.dumps(entry))

        if lines:
            with open(self.score_log_file, "a") as f:
                f.write("\n".join(lines) + "\n")

    def _log_epoch_losses(self, task_id, epoch, phase, phase_epoch, loss_meters, acc_meter):
        if not hasattr(self, "loss_log_file") or self.loss_log_file is None:
            return

        entry = {
            "task_id": task_id,
            "phase": phase,
            "epoch": epoch,
            "phase_epoch": phase_epoch,
            "total_loss": loss_meters["total"].avg,
            "ce_loss": loss_meters["ce"].avg,
            "prompt_loss": loss_meters["prompt"].avg,
            "router_loss": loss_meters["router"].avg,
            "proto_loss": loss_meters["proto"].avg,
            "attention_loss": loss_meters["attention"].avg,
            "train_acc": acc_meter.avg,
        }
        with open(self.loss_log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _save_snapshot(self, task_id):
        """Persist current pk, pv and old_q at the end of `task_id` so we can
        later compare attention outputs across tasks (e.g. PCA visualization).
        Captures a per-(layer, head) view; old_q here is the mean reduced-query
        of the just-finished task.
        """
        log_dir = self.config.get("log_dir", "outputs")
        snap_dir = os.path.join(log_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)

        pm = self._prompt_module()
        pk_all, pv_all, q_all = {}, {}, {}
        for e in pm.e_layers:
            for h in range(pm.num_heads):
                pk_eh, pv_eh = [], []
                for l in range(pm.num_experts):
                    pk_eh.append(getattr(pm, f"e_pk_{e}_{l}_{h}").detach().cpu().clone())
                    pv_eh.append(getattr(pm, f"e_pv_{e}_{l}_{h}").detach().cpu().clone())
                pk_all[(e, h)] = torch.cat(pk_eh, dim=0)  # (E, D)
                pv_all[(e, h)] = torch.cat(pv_eh, dim=0)  # (E, D)
                q_all[(e, h)] = getattr(pm, f"old_q_{e}_{h}").detach().cpu().clone()  # (1, D)

        snapshot = {
            "task_id": task_id,
            "pk": pk_all,
            "pv": pv_all,
            "query": q_all,
            "e_layers": list(pm.e_layers),
            "num_heads": pm.num_heads,
            "num_experts": pm.num_experts,
            "head_dim": pm.head_dim,
        }
        out_path = os.path.join(snap_dir, f"snapshot_task{task_id}.pt")
        torch.save(snapshot, out_path)
        print(f"Saved snapshot → {out_path}")

    def update_model(self, inputs, targets, dense=False):
        logits, prompt_loss = self.model(
            inputs, train=True, cls_mean=self.cls_mean, dense=dense
        )
        logits = logits[:, : self.valid_out_dim]

        logits[:, : self.last_valid_out_dim] = -float("inf")
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        ce_loss = self.criterion(logits, targets.long(), dw_cls)
        prompt_loss = prompt_loss.sum()
        total_loss = ce_loss + prompt_loss

        prompt_module = self._prompt_module()
        prompt_components = getattr(prompt_module, "last_loss_components", {})
        loss_components = {
            "total_loss": float(total_loss.detach().cpu().item()),
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "prompt_loss": float(prompt_loss.detach().cpu().item()),
            "router_loss": float(prompt_components.get("router_loss", 0.0)),
            "proto_loss": float(prompt_components.get("proto_loss", 0.0)),
            "attention_loss": float(prompt_components.get("attention_loss", 0.0)),
        }

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits, loss_components

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        task_id = self.task_count
        self.score_log_file = self._score_log_path(task_id)
        self.loss_log_file = self._loss_log_path(task_id)

        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        if self.reset_optimizer:
            self.log("Optimizer is reset!")
            self.init_optimizer()

        if need_train:
            if os.path.exists(self.score_log_file):
                os.remove(self.score_log_file)
            if os.path.exists(self.loss_log_file):
                os.remove(self.loss_log_file)

            self.data_weighting(train_dataset)
            batch_time = AverageMeter()

            if self.task_count == 0:
                print("-" * 10)
                print("Initial training...")
                self.log("Optimizer is reset!")
                epoch_factor = 0.5
                warmup_epochs = int(self.config["schedule"][-1] * epoch_factor)
                self.init_optimizer(epoch_factor=epoch_factor)
                self._learn_prompt_with_score_logging(
                    train_loader,
                    batch_time,
                    task_id,
                    dense=True,
                    epoch_factor=epoch_factor,
                    phase="warmup",
                    epoch_offset=0,
                )
                self.log("Optimizer is reset!")
                self.init_optimizer()
            else:
                warmup_epochs = 0

            self._learn_prompt_with_score_logging(
                train_loader,
                batch_time,
                task_id,
                phase="train",
                epoch_offset=warmup_epochs,
            )

            print("-" * 10)
            print("Updating expert usage statistics & collecting old queries...")
            prompt_module = self._prompt_module()
            feat_module = self._feat_module()
            e_layers = prompt_module.e_layers
            num_heads = prompt_module.num_heads
            head_dim = prompt_module.head_dim
            device = next(self.model.parameters()).device

            query_sums = {
                (e, h): torch.zeros(head_dim, device=device)
                for e in e_layers
                for h in range(num_heads)
            }
            query_count = 0

            num_samples = 0
            for i, (x, y, task) in enumerate(train_loader):
                self.model.eval()
                if self.gpu:
                    x = x.cuda()
                with torch.no_grad():
                    prompt_scores = self.model(x, train=False, return_attn=True)
                self.model.prompt.update_prompt(prompt_scores)
                num_samples += x.size(0)

                # Accumulate mean reduced-query per (layer, head)
                q_prompts = feat_module.last_q_prompts
                for e in e_layers:
                    if e >= len(q_prompts) or q_prompts[e] is None:
                        continue
                    q = q_prompts[e]  # (B, num_heads, 1, head_dim)
                    for h in range(num_heads):
                        query_sums[(e, h)] += q[:, h, 0, :].sum(dim=0)
                query_count += x.size(0)

            self.model.prompt.update_num_samples(num_samples)

            if query_count > 0:
                query_means = {
                    k: v / query_count for k, v in query_sums.items()
                }
                prompt_module.save_old_queries(query_means)

            self._save_snapshot(task_id)
            print("-" * 10)

        self.model.eval()
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(
                self.memory_size, np.arange(self.last_valid_out_dim)
            )

        try:
            return batch_time.avg, need_train
        except:
            return None, need_train

    def _learn_prompt_with_score_logging(
        self,
        train_loader,
        batch_time,
        task_id,
        dense=False,
        epoch_factor=1,
        phase="train",
        epoch_offset=0,
    ):
        batch_timer = Timer()
        num_epochs = int(self.config["schedule"][-1] * epoch_factor)

        if self.schedule_type == "coswm":
            for phase_epoch in range(num_epochs):
                self.epoch = phase_epoch
                epoch = epoch_offset + phase_epoch
                losses = {
                    "total": AverageMeter(),
                    "ce": AverageMeter(),
                    "prompt": AverageMeter(),
                    "router": AverageMeter(),
                    "proto": AverageMeter(),
                    "attention": AverageMeter(),
                }
                acc = AverageMeter()
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):
                    self.model.train()
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    loss, output, loss_components = self.update_model(x, y, dense=dense)
                    self.scheduler.step()
                    self._log_batch_scores(
                        self._feat_module().last_prompt_scores,
                        task_id,
                        epoch,
                        phase,
                        phase_epoch,
                        i,
                    )

                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses["total"].update(loss_components["total_loss"], y.size(0))
                    losses["ce"].update(loss_components["ce_loss"], y.size(0))
                    losses["prompt"].update(loss_components["prompt_loss"], y.size(0))
                    losses["router"].update(loss_components["router_loss"], y.size(0))
                    losses["proto"].update(loss_components["proto_loss"], y.size(0))
                    losses["attention"].update(loss_components["attention_loss"], y.size(0))
                    batch_timer.tic()

                self._log_epoch_losses(task_id, epoch, phase, phase_epoch, losses, acc)
                self.log(
                    "Epoch:{epoch:.0f}/{total:.0f}".format(
                        epoch=phase_epoch + 1, total=num_epochs
                    ),
                    end=" ",
                )
                self.log(
                    " * Total {total.avg:.3f} | CE {ce.avg:.3f} | Router {router.avg:.3f} | Proto {proto.avg:.3f} | Attn {attention.avg:.3f} | Train Acc {acc.avg:.3f}".format(
                        total=losses["total"],
                        ce=losses["ce"],
                        router=losses["router"],
                        proto=losses["proto"],
                        attention=losses["attention"],
                        acc=acc,
                    )
                )
        else:
            for phase_epoch in range(num_epochs):
                self.epoch = phase_epoch
                epoch = epoch_offset + phase_epoch
                if phase_epoch > 0:
                    self.scheduler.step()
                losses = {
                    "total": AverageMeter(),
                    "ce": AverageMeter(),
                    "prompt": AverageMeter(),
                    "router": AverageMeter(),
                    "proto": AverageMeter(),
                    "attention": AverageMeter(),
                }
                acc = AverageMeter()
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):
                    self.model.train()
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    loss, output, loss_components = self.update_model(x, y, dense=dense)
                    self._log_batch_scores(
                        self._feat_module().last_prompt_scores,
                        task_id,
                        epoch,
                        phase,
                        phase_epoch,
                        i,
                    )

                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses["total"].update(loss_components["total_loss"], y.size(0))
                    losses["ce"].update(loss_components["ce_loss"], y.size(0))
                    losses["prompt"].update(loss_components["prompt_loss"], y.size(0))
                    losses["router"].update(loss_components["router_loss"], y.size(0))
                    losses["proto"].update(loss_components["proto_loss"], y.size(0))
                    losses["attention"].update(loss_components["attention_loss"], y.size(0))
                    batch_timer.tic()

                self._log_epoch_losses(task_id, epoch, phase, phase_epoch, losses, acc)
                self.log(
                    "Epoch:{epoch:.0f}/{total:.0f}".format(
                        epoch=phase_epoch + 1, total=num_epochs
                    )
                )
                self.log(
                    " * Total {total.avg:.3f} | CE {ce.avg:.3f} | Router {router.avg:.3f} | Proto {proto.avg:.3f} | Attn {attention.avg:.3f} | Train Acc {acc.avg:.3f}".format(
                        total=losses["total"],
                        ce=losses["ce"],
                        router=losses["router"],
                        proto=losses["proto"],
                        attention=losses["attention"],
                        acc=acc,
                    )
                )


class BaselineSMoPELearner(VAPTPrompt):
    """
    Baseline variant of VAPTPrompt learner:
    - Uses BaselineSMoPEPrompt (no top-K routing, no auxiliary losses)
    - Logs per-expert, per-layer, per-head attention scores to
      <log_dir>/expert_scores_task<t>.jsonl after each training epoch
    """

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim,
            prompt_flag="baseline_smope",
            prompt_param=self.prompt_param,
            pretrained=cfg["pretrained_weight"],
        )
        return model

    def _score_log_path(self, task_id):
        log_dir = self.config.get("log_dir", "outputs")
        score_dir = os.path.join(log_dir, "expert_scores")
        os.makedirs(score_dir, exist_ok=True)
        return os.path.join(score_dir, f"expert_scores_task{task_id}.jsonl")

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        # Set log file path on the prompt module
        task_id = self.task_count
        self.model.prompt.set_score_log_file(self._score_log_path(task_id))

        # Try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        if self.reset_optimizer:
            self.log("Optimizer is reset!")
            self.init_optimizer()

        if need_train:
            self.data_weighting(train_dataset)
            batch_time = AverageMeter()

            if self.task_count == 0:
                print("-" * 10)
                print("Initial training...")
                self.log("Optimizer is reset!")
                epoch_factor = 0.5
                self.init_optimizer(epoch_factor=epoch_factor)
                self._learn_prompt_with_score_logging(
                    train_loader, batch_time, task_id, dense=True, epoch_factor=epoch_factor
                )
                self.log("Optimizer is reset!")
                self.init_optimizer()

            self._learn_prompt_with_score_logging(train_loader, batch_time, task_id)

            print("-" * 10)
            print("Logging expert scores on full train set...")
            num_samples = 0
            for i, (x, y, task) in enumerate(train_loader):
                self.model.eval()
                if self.gpu:
                    x = x.cuda()
                with torch.no_grad():
                    prompt_scores = self.model(x, train=False, return_attn=True)
                # Log scores (epoch=-1 means post-training pass)
                self.model.prompt.log_scores(prompt_scores, task_id, batch_idx=i, epoch=-1)
                num_samples += x.size(0)
            self.model.prompt.update_num_samples(num_samples)
            print("-" * 10)

        self.model.eval()
        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(
                self.memory_size, np.arange(self.last_valid_out_dim)
            )

        try:
            return batch_time.avg, need_train
        except:
            return None, need_train

    def _learn_prompt_with_score_logging(
        self, train_loader, batch_time, task_id, dense=False, epoch_factor=1
    ):
        losses = AverageMeter()
        acc = AverageMeter()
        batch_timer = Timer()
        num_epochs = int(self.config["schedule"][-1] * epoch_factor)

        if self.schedule_type == "coswm":
            for epoch in range(num_epochs):
                self.epoch = epoch
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):
                    self.model.train()
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    loss, output = self.update_model(x, y, dense=dense)
                    self.scheduler.step()

                    # Log scores every epoch on first batch for lightweight tracking
                    if i == 0:
                        self.model.eval()
                        with torch.no_grad():
                            prompt_scores = self.model(x, train=False, return_attn=True)
                        self.model.prompt.log_scores(prompt_scores, task_id, i, epoch)
                        self.model.train()

                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                self.log(
                    "Epoch:{epoch:.0f}/{total:.0f}".format(epoch=self.epoch + 1, total=num_epochs),
                    end=" ",
                )
                self.log(
                    " * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}".format(loss=losses, acc=acc)
                )
                losses = AverageMeter()
                acc = AverageMeter()
        else:
            for epoch in range(num_epochs):
                self.epoch = epoch
                if epoch > 0:
                    self.scheduler.step()
                batch_timer.tic()
                for i, (x, y, task) in enumerate(train_loader):
                    self.model.train()
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()

                    loss, output = self.update_model(x, y, dense=dense)

                    # Log scores every epoch on first batch for lightweight tracking
                    if i == 0:
                        self.model.eval()
                        with torch.no_grad():
                            prompt_scores = self.model(x, train=False, return_attn=True)
                        self.model.prompt.log_scores(prompt_scores, task_id, i, epoch)
                        self.model.train()

                    batch_time.update(batch_timer.toc())
                    batch_timer.tic()
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss, y.size(0))
                    batch_timer.tic()

                self.log(
                    "Epoch:{epoch:.0f}/{total:.0f}".format(epoch=self.epoch + 1, total=num_epochs)
                )
                self.log(
                    " * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}".format(loss=losses, acc=acc)
                )
                losses = AverageMeter()
                acc = AverageMeter()


# @inproceedings{smith2023coda,
#   title={CODA-Prompt: COntinual decomposed attention-based prompting for rehearsal-free continual learning},
#   author={Smith, James Seale and Karlinsky, Leonid and Gutta, Vyshnavi and Cascante-Bonilla, Paola and Kim, Donghyun and Arbelle, Assaf and Panda, Rameswar and Feris, Rogerio and Kira, Zsolt},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={11909--11919},
#   year={2023}
# }
class CODAPrompt(Prompt):
    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim, prompt_flag="coda", prompt_param=self.prompt_param
        )
        return model


# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):
    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim, prompt_flag="dual", prompt_param=self.prompt_param
        )
        return model


# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):
    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg["model_type"]].__dict__[cfg["model_name"]](
            out_dim=self.out_dim, prompt_flag="l2p", prompt_param=self.prompt_param
        )
        return model

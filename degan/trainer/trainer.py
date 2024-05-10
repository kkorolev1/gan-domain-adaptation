import random
from pathlib import Path
from random import shuffle
import itertools
from glob import glob
from collections import defaultdict

from numpy import inf
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
from torchvision.transforms import v2
from torchvision.utils import make_grid

from degan.utils import inf_loop, MetricTracker, ten2img
from degan.logger import get_visualizer

class Trainer:
    """
    Trainer class
    """

    def __init__(
            self,
            generator,
            domain_encoder,
            clip_encoder,
            criterion,
            metrics,
            optimizer_encoder,
            lr_scheduler_encoder,
            config,
            device,
            dataloaders,
            ema=None,
            len_epoch=None,
            skip_oom=True
    ):
        self.generator = generator
        self.domain_encoder = domain_encoder
        self.clip_encoder = clip_encoder

        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.criterion = criterion
        self.metrics = metrics
        self.optimizer_encoder = optimizer_encoder
        self.lr_scheduler_encoder = lr_scheduler_encoder

        # for interrupt saving
        self._last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")
        self.reset_optimizer = cfg_trainer.get("reset_optimizer", False)
        self.reset_scheduler = cfg_trainer.get("reset_scheduler", False)

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = get_visualizer(
            config, self.logger, cfg_trainer["visualize"]
        )

        self.skip_oom = skip_oom
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *self.criterion.loss_names, writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *self.criterion.loss_names,
            *[m.name for m in self.metrics], writer=self.writer
        )
    
        self.use_ema = ema is not None
        self.ema = ema

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self.mnt_mode == "min":
                        improved = log[self.mnt_metric] <= self.mnt_best
                    elif self.mnt_mode == "max":
                        improved = log[self.mnt_metric] >= self.mnt_best
                    else:
                        improved = False
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            "epoch": epoch,
            "state_dict_encoder": self.domain_encoder.state_dict(),
            "optimizer_encoder": self.optimizer_encoder.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        if "warmup_scheduler" in self.config:
            state.update({
                "lr_scheduler_encoder": self.lr_scheduler_encoder.lr_scheduler.state_dict(),
                "warmup_lr_scheduler_encoder": self.lr_scheduler_encoder.state_dict()
            })
        else:
            state.update({
                "lr_scheduler_encoder": self.lr_scheduler_encoder.state_dict()
            })
        if self.use_ema:
            state.update({
                "ema": self.ema.state_dict()
            })
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            # for ckpt_path in glob(str(self.checkpoint_dir / "checkpoint-epoch*.pth")):
            #     os.remove(ckpt_path)
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        self.domain_encoder.load_state_dict(checkpoint["state_dict_encoder"])

        if not self.reset_optimizer:
            self.logger.info("Loading optimizer state")
            self.optimizer_encoder.load_state_dict(checkpoint["optimizer_encoder"])

        if not self.reset_scheduler:
            self.logger.info("Loading scheduler state")
            if "warmup_scheduler" in self.config:
                self.lr_scheduler_encoder.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_encoder"])
                self.lr_scheduler_encoder.load_state_dict(checkpoint["warmup_lr_scheduler_encoder"])
            else:
                self.lr_scheduler_encoder.load_state_dict(checkpoint["lr_scheduler_encoder"])
            

        if self.use_ema:
            self.ema.load_state_dict(checkpoint["ema"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["domain_img", "inversion_img", "latent"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, model):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.eval()
        self.domain_encoder.train()
        self.clip_encoder.eval()

        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        last_train_metrics = self.train_metrics.result()

        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    batch_idx,
                    metrics=self.train_metrics,
                    is_train=True
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.domain_encoder.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx > 0 and batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler_encoder.get_last_lr()[0]
                )
                if batch_idx % (self.len_epoch // 2) == 0:
                    self._log_predictions(**batch)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log


    def process_batch(self, batch, batch_idx, metrics: MetricTracker, is_train=True, use_ema_encoder=False):
        batch = self.move_batch_to_device(batch, self.device)
        domain_img = batch["domain_img"]
        inversion_img = batch["inversion_img"]
        latent = batch["latent"]

        if is_train or not use_ema_encoder:
            domain_encoder = self.domain_encoder
        else:
            domain_encoder = self.ema

        domain_chunks = domain_encoder(domain_img)
        batch["domain_offset"] = torch.cat(domain_chunks, dim=1)
        batch["gen_img"] = self.generator([latent], domain_chunks=domain_chunks)[0]
        batch["src_img"] = self.generator([latent])[0]

        batch["gen_emb"] = self.clip_encoder.encode_img(batch["gen_img"])
        batch["src_emb"] = self.clip_encoder.encode_img(batch["src_img"])
        batch["domain_emb"] = self.clip_encoder.encode_img(domain_img)
        batch["src_emb_proj"] = self.clip_encoder.encode_img(inversion_img)

        loss_dict = self.criterion(**batch)
        for key, loss_value in loss_dict.items():
            loss_dict[key] = loss_value / self.config["trainer"]["grad_accumulation_steps"]
        loss = loss_dict["loss"]
    
        if is_train:
            loss.backward()
            if ((batch_idx + 1) % self.config["trainer"]["grad_accumulation_steps"] == 0) or (batch_idx + 1 == self.len_epoch):
                self._clip_grad_norm(self.domain_encoder)
                self.optimizer_encoder.step()
                self.lr_scheduler_encoder.step()
                metrics.update("grad norm", self.get_grad_norm(self.domain_encoder))
                if self.use_ema:
                    self.ema.update()
                self.optimizer_encoder.zero_grad()
        
        batch.update(loss_dict)
        for key, loss_value in loss_dict.items():
            metrics.update(key, batch[key].item())

        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.domain_encoder.eval()
        if self.use_ema:
            self.ema.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            self.writer.set_step(epoch * self.len_epoch, part)
            
            keys = ["domain_img", "gen_img", "src_img", "domain_path", "gen_emb", "domain_emb"]
            batch_dict = {key: [] for key in keys}

            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    batch_idx,
                    metrics=self.evaluation_metrics,
                    is_train=False,
                    use_ema_encoder=(part == "val_ema")
                )
                for key in keys:
                    items = batch[key]
                    if isinstance(items, torch.Tensor):
                        # move to cpu because of the OOM
                        items = items.cpu()
                    batch_dict[key].append(items)

            # Group by domain_path, because metrics are calculated per domain

            for key in ["domain_img", "gen_img", "src_img", "gen_emb", "domain_emb"]:
                # move to cpu because of the OOM
                batch_dict[key] = torch.cat(batch_dict[key]).cpu()
            batch_dict["domain_path"] = list(itertools.chain(*batch_dict["domain_path"]))
            domain_to_gen_emb = defaultdict(list)
            domain_to_domain_emb = defaultdict(list)

            for domain_path, gen_emb, domain_emb in zip(batch_dict["domain_path"], batch_dict["gen_emb"], batch_dict["domain_emb"]):
                domain_to_gen_emb[domain_path].append(gen_emb.unsqueeze(0))
                domain_to_domain_emb[domain_path] = domain_emb

            for domain_path in domain_to_gen_emb:
                domain_to_gen_emb[domain_path] = torch.cat(domain_to_gen_emb[domain_path], dim=0).to(self.device)
                domain_to_domain_emb[domain_path] = domain_to_domain_emb[domain_path].to(self.device)

            for met in self.metrics:
                self.evaluation_metrics.update(met.name, met(domain_to_gen_emb, domain_to_domain_emb))

            self._log_predictions(**batch_dict)
            self._log_scalars(self.evaluation_metrics)

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
            #self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()
    
    def _log_predictions(
            self,
            domain_img,
            gen_img,
            src_img,
            examples_to_log=15,
            *args,
            **kwargs
    ):
        if self.writer is None:
            return
        domain_img = domain_img.clip(-1, 1)
        gen_img = gen_img.clip(-1, 1)
        src_img = src_img.clip(-1, 1)
        transform = v2.Compose([
            v2.ToPILImage()
        ])

        indices = list(range(domain_img.shape[0]))
        shuffle(indices)
        indices = indices[:examples_to_log]

        domain_img = domain_img[indices]
        src_img = src_img[indices]
        gen_img = gen_img[indices]

        stack = torch.cat([domain_img, src_img, gen_img], dim=0)
        grid = make_grid(stack, nrow=domain_img.shape[0], normalize=True, value_range=(-1, 1))
        image = transform(grid.cpu())

        self.writer.add_image("grid", image)
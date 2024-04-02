import random
from pathlib import Path
from random import shuffle
import itertools
from glob import glob

from numpy import inf
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os


from DEGAN.utils import inf_loop, MetricTracker

from DEGAN.logger import get_visualizer

class Trainer:
    """
    Trainer class
    """

    def __init__(
            self,
            generator,
            encoder,
            criterion,
            metrics,
            optimizer_encoder,
            lr_scheduler_encoder,
            config,
            device,
            dataloaders,
            len_epoch=None,
            skip_oom=True
    ):
        self.generator = generator
        self.encoder = encoder

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
        self.config = config
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
            "g_loss",
            "g_grad_norm",
            *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

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
            "state_dict_encoder": self.encoder.state_dict(),
            "optimizer_encoder": self.optimizer_encoder.state_dict(),
            "lr_scheduler_encoder": self.lr_scheduler_encoder.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
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

        self.encoder.load_state_dict(checkpoint["state_dict_encoder"])

        if not self.reset_optimizer:
            self.logger.info("Loading optimizer state")
            self.optimizer_encoder.load_state_dict(checkpoint["optimizer_encoder"])

        if not self.reset_scheduler:
            self.logger.info("Loading scheduler state")
            self.lr_scheduler_encoder.load_state_dict(checkpoint["lr_scheduler_encoder"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["wav_gt", "mel_gt"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, module):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                module.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.encoder.train()

        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    batch_idx,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} DLoss: {:.6f} GLoss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["d_loss"].item(),
                        batch["g_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "discriminator learning rate", self.lr_scheduler_d.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "generator learning rate", self.lr_scheduler_g.get_last_lr()[0]
                )
                self._log_predictions(**batch, is_train=True)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        self.lr_scheduler_d.step()
        self.lr_scheduler_g.step()

        return log


    def process_batch(self, batch, batch_idx, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        wav_gt = batch["wav_gt"]
        mel_spec_gt = batch["mel_gt"]

        wav_pred = self.model.generator(mel_spec_gt)
        batch["wav_pred"] = wav_pred
        mel_spec_pred = self.mel_spec_transform(wav_pred).squeeze(1)

        # ---- Discriminator loss
        self.optimizer_d.zero_grad()

        # Don't need features for discriminator loss
        mpd_gt_outputs, _ = self.model.mpd(wav_gt)
        mpd_outputs, _ = self.model.mpd(wav_pred.detach())

        msd_gt_outputs, _ = self.model.msd(wav_gt)
        msd_outputs, _ = self.model.msd(wav_pred.detach())

        mpd_d_loss = self.criterion.discriminator_adv_loss(mpd_gt_outputs, mpd_outputs)
        msd_d_loss = self.criterion.discriminator_adv_loss(msd_gt_outputs, msd_outputs)

        d_loss = mpd_d_loss + msd_d_loss

        d_loss.backward()
        # TODO: clip_grad_norm
        self._clip_grad_norm(self.model.mpd)
        self._clip_grad_norm(self.model.msd)
        self.optimizer_d.step()

        batch["mpd_d_loss"] = mpd_d_loss
        batch["msd_d_loss"] = msd_d_loss
        batch["d_loss"] = d_loss
        
        d_params = itertools.chain(self.model.mpd.parameters(), self.model.msd.parameters())
        batch["d_grad_norm"] = torch.tensor([self.get_grad_norm(d_params)])

        # ---- Generator loss
        self.optimizer_g.zero_grad()

        # Don't need gt output for generator loss
        _, mpd_gt_features = self.model.mpd(wav_gt)
        mpd_outputs, mpd_features = self.model.mpd(wav_pred)

        _, msd_gt_features = self.model.msd(wav_gt)
        msd_outputs, msd_features = self.model.msd(wav_pred)

        mpd_g_loss = self.criterion.generator_adv_loss(mpd_outputs)
        msd_g_loss = self.criterion.generator_adv_loss(msd_outputs)

        mel_spec_g_loss = self.criterion.mel_spectrogram_loss(mel_spec_gt, mel_spec_pred)
        
        mpd_features_g_loss = self.criterion.feature_matching_loss(mpd_gt_features, mpd_features)
        msd_features_g_loss = self.criterion.feature_matching_loss(msd_gt_features, msd_features)

        g_loss = mpd_g_loss + msd_g_loss + mel_spec_g_loss + mpd_features_g_loss + msd_features_g_loss

        g_loss.backward()
        # TODO: clip_grad_norm
        self._clip_grad_norm(self.model.generator)
        self.optimizer_g.step()

        batch["mpd_g_loss"] = mpd_g_loss
        batch["msd_g_loss"] = msd_g_loss
        batch["mel_spec_g_loss"] = mel_spec_g_loss
        batch["mpd_features_g_loss"] = mpd_features_g_loss
        batch["msd_features_g_loss"] = msd_features_g_loss
        batch["g_loss"] = g_loss
        g_params = self.model.generator.parameters()
        batch["g_grad_norm"] = torch.tensor([self.get_grad_norm(g_params)])
    
        for metric_key in metrics.keys():
            metrics.update(metric_key, batch[metric_key].item())

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
    def get_grad_norm(self, parameters, norm_type=2):
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
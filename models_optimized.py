import os
from typing import Any, Dict, Optional, Union

import torch
import torch.optim as optim
import lightning as L
import pandas as pd
from transformers import AutoModelForVideoClassification, AutoConfig
from torchmetrics import Accuracy, F1Score, Precision, Recall
from pytorchvideo.transforms import MixUp


class VideoModel(L.LightningModule):
    """
    PyTorch Lightning Module for video classification using a pretrained model
    from Hugging Face Transformers. Supports optional freezing of model parameters,
    mixup augmentation, and configurable optimizers/schedulers.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        cache_dir: Optional[str] = None,
        lr: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the VideoModel.

        Parameters:
            model_name: Name or path of the pretrained model.
            num_classes: Number of target classes.
            cache_dir: Directory to cache the pretrained model.
            lr: Learning rate.
            optimizer: Choice of optimizer ('adamw', 'adam', 'sgd').
            scheduler: Scheduler type ('plateau', 'step', 'linearlr', 'cosine') or None.
            args: Additional arguments as a dictionary. Optionally includes:
                  - no_pretrain: Whether to initialize model from config.
                  - freeze: Whether to freeze the backbone.
                  - mixup: Whether to apply mixup augmentation.
                  - mixup_alpha: Alpha value for mixup.
                  - use_amp: If True, wrap forward pass with torch.cuda.amp.autocast().
                  - exp_name: Experiment name for logging.
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.cache_dir = cache_dir
        self.lr = lr
        self.optimizer_name = optimizer
        self.scheduler_type = scheduler
        self.args = args or {}
        self.use_amp = self.args.get("use_amp", False)

        # Load the model: either from scratch via config or from a pretrained checkpoint.
        if self.args.get("no_pretrain", False):
            config = AutoConfig.from_pretrained(
                model_name, num_labels=num_classes, local_files_only=False
            )
            self.model = AutoModelForVideoClassification.from_config(config)
        else:
            self.model = AutoModelForVideoClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=True,
            )

        # Freeze backbone parameters if specified.
        if self.args.get("freeze", False):
            self._freeze_backbone()

        # Initialize mixup if requested.
        if self.args.get("mixup", False):
            alpha = self.args.get("mixup_alpha", 1.0)
            self.mixup = MixUp(alpha=alpha, num_classes=num_classes)

        # Initialize training metrics.
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.f1_train = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # Initialize validation metrics.
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.f1_val = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.topk_val_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro", top_k=5
        )

        self.save_hyperparameters()

    def _freeze_backbone(self) -> None:
        """
        Freeze all model parameters except for those in the classifier.
        Supports different model architectures.
        """
        if hasattr(self.model, "classifier"):
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "model") and hasattr(self.model.model, "classifier"):
            for param in self.model.model.parameters():
                param.requires_grad = False
            for param in self.model.model.classifier.parameters():
                param.requires_grad = True
        else:
            self.print("Warning: Could not freeze backbone properly; model structure is unexpected.")

    def load_pretrained_weights(self, checkpoint_path: str) -> None:
        """
        Load pretrained weights from a checkpoint file.

        Parameters:
            checkpoint_path: Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint.get("state_dict", {})
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.", "")
            new_state_dict[new_key] = v
        self.model.load_state_dict(new_state_dict, strict=False)


    def forward(self, x: torch.Tensor) -> Any:
        print("Input shape to forward:", x.shape)  # Should be (B, C, T, H, W)
        # Permute once: convert from (B, C, T, H, W) to (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        print("After permute:", x.shape)  # Should be (B, T, C, H, W)
        return self.model(x)


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step: compute loss, update metrics, and log values.
        """
        x, y = batch[0], batch[1]
        # Optionally use AMP.
        if self.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.get("mixup", False) and x.shape[0] > 1:
                    x, y_soft = self.mixup(x, y)
                    outputs = self(x)
                    loss = torch.nn.functional.cross_entropy(outputs.logits, y_soft)
                else:
                    outputs = self(x, labels=y)
                    loss = outputs.loss
        else:
            if self.args.get("mixup", False) and x.shape[0] > 1:
                x, y_soft = self.mixup(x, y)
                outputs = self(x)
                loss = torch.nn.functional.cross_entropy(outputs.logits, y_soft)
            else:
                outputs = self(x, labels=y)
                loss = outputs.loss

        # Update metrics.
        self.train_acc.update(outputs.logits, y)
        self.f1_train.update(outputs.logits, y)

        # Log metrics using a single log_dict call.
        self.log_dict({
            "train_loss": loss,
            "train_acc": self.train_acc,
            "f1_train": self.f1_train
        }, sync_dist=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Validation step: compute loss, update metrics, and log validation values.
        """
        x, y = batch[0], batch[1]
        # Optionally use AMP.
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self(x, labels=y)
                loss = outputs.loss
        else:
            outputs = self(x, labels=y)
            loss = outputs.loss

        # Update validation metrics.
        self.val_acc.update(outputs.logits, y)
        self.f1_val.update(outputs.logits, y)
        self.recall.update(outputs.logits, y)
        self.precision.update(outputs.logits, y)
        self.topk_val_acc.update(outputs.logits, y)

        # Log validation metrics.
        self.log_dict({
            "val_loss": loss,
            "val_acc": self.val_acc,
            "top5_val_acc": self.topk_val_acc,
            "f1_val": self.f1_val,
            "recall": self.recall,
            "precision": self.precision,
        }, sync_dist=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> Union[optim.Optimizer, Dict[str, Any]]:
        """
        Configure the optimizer and (optionally) learning rate scheduler.
        """
        if self.optimizer_name == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Invalid optimizer choice: " + self.optimizer_name)

        if self.scheduler_type:
            if self.scheduler_type == "plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.1, patience=20, verbose=True
                )
            elif self.scheduler_type == "step":
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
            elif self.scheduler_type == "linearlr":
                scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.333, end_factor=1, total_iters=100)
            elif self.scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
            else:
                raise ValueError("Invalid scheduler type: " + self.scheduler_type)

            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

        return optimizer

    def on_train_epoch_end(self) -> None:
        """
        Callback at the end of each training epoch.
        (Currently unused but can be extended as needed.)
        """
        pass

    def on_validation_epoch_end(self) -> None:
        """
        At the end of a validation epoch, compute metrics and save results to a CSV file.
        """
        exp_name = self.hparams["args"].get("exp_name", "experiment")
        experiment_folder = os.path.join("lightning_logs", exp_name)
        os.makedirs(experiment_folder, exist_ok=True)

        metrics_data = {
            "model_name": exp_name,
            "val_acc": self.val_acc.compute().item(),
            "top5_val_acc": self.topk_val_acc.compute().item(),
            "f1_val": self.f1_val.compute().item(),
            "recall": self.recall.compute().item(),
            "precision": self.precision.compute().item(),
        }

        save_path = os.path.join(experiment_folder, f"{exp_name}_validation_results.csv")
        metrics_df = pd.DataFrame([metrics_data])
        metrics_df.to_csv(save_path, index=False)
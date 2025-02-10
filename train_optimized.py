import os
import random
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from dataset_optimized import DatasetFactory
from transforms_optimized import VideoTransforms
from torch.utils.data import DataLoader
from models_optimized import VideoModel
from argparse import ArgumentParser
from utils import set_seed
import yaml  # For saving hyperparameters to hparams.yaml

# Set some global environment variables if needed
os.environ["NCCL_P2P_DISABLE"] = "1"
torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    parser = ArgumentParser()

    # Experiment parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers for training")
    # parser.add_argument("--val_workers", type=int, default=4, help="Number of DataLoader workers for validation")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Path to cache directory for pretrained models")
    parser.add_argument("--no_pretrain", action="store_true", help="Initialize model from config (no pretrained weights)")
    parser.add_argument("--finetune", type=str, default=None, help="Path to checkpoint for finetuning")
    parser.add_argument("--freeze", action="store_true", help="Freeze backbone parameters")
    
    # Model and training parameters
    parser.add_argument("-ptm", "--pretrained_model", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics", 
                        help="Pretrained model", 
                        choices=["MCG-NJU/videomae-base-finetuned-kinetics", "google/vivit-b-16x2-kinetics400", "facebook/timesformer-base-finetuned-k400"])
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-epochs", "--max_epochs", type=int, default=200, help="Maximum number of epochs")
    parser.add_argument("-gpus", "--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("-opt", "--optimizer", type=str, default="adamw", help="Optimizer")
    parser.add_argument("-sched", "--scheduler", type=str, default=None, help="Scheduler type")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="../MINDS_tensors_32", help="Root directory of the dataset")
    parser.add_argument("--specific_classes", type=str, nargs="+", default=None, help="List of specific classes to use")
    parser.add_argument("--dataset", type=str, default="minds", help="Dataset name")
    parser.add_argument("--n_samples_per_class", type=int, default=None, help="Limit samples per class")
    
    # Transforms parameters
    parser.add_argument("-t", "--transforms", type=str, nargs="+", default=[], help="List of augmentation names")
    parser.add_argument("-tp", "--transforms_parameters", type=str, nargs="+", default=[], help="List of augmentation parameters")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to sample (16 or 32)")
    parser.add_argument("--mixup", action="store_true", help="Enable mixup augmentation")
    parser.add_argument("--mixup_alpha", type=float, default=1.0, help="Mixup alpha value")
    parser.add_argument("--use_kinetics_norm", action="store_true", default=False, help="Use Kinetics normalization instead of dataset-specific normalization")
    parser.add_argument("--patience", type=int, default=30, help="Patience (in epochs) for early stopping (default: 30)")
    
    args = parser.parse_args()
    args_dict = vars(args)
    
    # Validate frame count
    if args.frames not in [16, 32]:
        raise ValueError("Frames must be either 16 or 32")
    if args.frames == 16 and args.pretrained_model == "google/vivit-b-16x2-kinetics400":
        raise ValueError("Vivit model only supports 32 frames")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set experiment name if not provided
    if args.exp_name:
        EXP_NAME = args.exp_name
    else:
        EXP_NAME = args.pretrained_model.replace("/", "-")
    
    # Save hyperparameters to a YAML file inside the experiment folder
    exp_folder = os.path.join("lightning_logs", EXP_NAME)
    os.makedirs(exp_folder, exist_ok=True)
    with open(os.path.join(exp_folder, "hparams.yaml"), "w") as f:
        yaml.dump(args_dict, f)
    
    # Create transforms using the optimized VideoTransforms class.
    train_transforms = VideoTransforms(
        mode="train",
        input_size=(224, 224),
        num_frames=args.frames,
        use_kinetics_norm=args.use_kinetics_norm,
        transforms_list=args.transforms,
        transforms_parameters=args.transforms_parameters
    )
    val_transforms = VideoTransforms(
        mode="eval",
        input_size=(224, 224),
        num_frames=args.frames,
        use_kinetics_norm=args.use_kinetics_norm,
        transforms_list=["normalize"],
    )
    
    # Instantiate datasets using DatasetFactory.
    dataset_factory = DatasetFactory()
    train_dataset = dataset_factory(
        name=args.dataset,
        root_dir=args.data_path,
        transform=train_transforms,
        split="train",
        specific_classes=args.specific_classes,
        n_samples_per_class=args.n_samples_per_class,
    )
    
    val_dataset = dataset_factory(
        name=args.dataset,
        root_dir=args.data_path,
        transform=val_transforms,
        split="test",
        specific_classes=args.specific_classes,
    )
    
    # Create DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    # Set up callbacks and logger.
    checkpoint_callback = ModelCheckpoint(
        filename="best_model",
        save_top_k=1,
        save_last=False,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    top5_checkpoint_callback = ModelCheckpoint(
        filename="top5_best_model",
        save_top_k=1,
        save_last=False,
        verbose=False,
        monitor="top5_val_acc",
        mode="max",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        verbose=False,
        mode="min",
    )
    
    logger = TensorBoardLogger(save_dir="lightning_logs", name=EXP_NAME)
    
    # Create the model using the optimized models.py.
    model = VideoModel(
        args.pretrained_model,
        num_classes=len(train_dataset.classes),
        cache_dir=args.cache_dir,
        lr=args.learning_rate,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        args=args_dict,
    )
    
    # If a finetuning checkpoint is provided, load it and adjust the classifier.
    if args.finetune:
        print("=" * 30)
        print("Finetuning model...")
        model.load_pretrained_weights(args.finetune)
        classifier_in_features = model.model.classifier.in_features
        model.model.classifier = torch.nn.Linear(classifier_in_features, len(train_dataset.classes))
        print("Adjusted classifier to", len(train_dataset.classes), "classes")
        print("=" * 30)
    
    # Create the Lightning Trainer.
    trainer = L.Trainer(
        log_every_n_steps=20,
        max_epochs=args.max_epochs,
        devices=args.gpus,
        accelerator="gpu",
        strategy="ddp",
        callbacks=[checkpoint_callback, top5_checkpoint_callback, early_stop_callback],
        logger=logger,
    )
    
    # Start training.
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

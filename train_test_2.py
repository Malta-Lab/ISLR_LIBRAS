import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from dataset_test_2 import DatasetFactory
# from transforms import build_transforms
from transforms import Transforms
from torch.utils.data import DataLoader
from models import VideoModel
import torch
import os
from argparse import ArgumentParser
from utils import set_seed

os.environ["NCCL_P2P_DISABLE"] = "1"
torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    parser = ArgumentParser()
    # experiment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument('--freeze', action='store_true')
    
    # model and training
    parser.add_argument(
        "-ptm", "--pretrained_model", type=str, default="MCG-NJU/videomae-base-finetuned-kinetics", help="Pretrained model", choices=["MCG-NJU/videomae-base-finetuned-kinetics", "google/vivit-b-16x2-kinetics400", "facebook/timesformer-base-finetuned-k400"]
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-epochs", "--max_epochs", type=int, default=200)
    parser.add_argument("-gpus", "--gpus", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-opt", "--optimizer", type=str, default="adamw")
    parser.add_argument("-sched", "--scheduler", type=str, default=None)
    
    # data 
    parser.add_argument("--data_path", type=str, default="../MINDS_tensors")
    parser.add_argument("--specific_classes", type=str, nargs="+", default=None)
    parser.add_argument("--dataset", type=str, default="minds")
    parser.add_argument("--n_samples_per_class", type=int, default=None)
    
    # transforms
    parser.add_argument("-t", "--transforms", type=str, nargs="+", default=None)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("-tp", "--transforms_parameters", type=str, nargs="+", default=None)
    parser.add_argument("--mixup_alpha", type=float, default=1.0)

    args = parser.parse_args()

    args_dict = vars(args)


    if args.frames != 32 and args.frames != 16:
        raise ValueError("Frames must be 16 or 32")
    if args.frames == 16 and args.pretrained_model == "google/vivit-b-16x2-kinetics400":
        raise ValueError("Vivit model only supports 32 frames")        

    set_seed(args.seed)

    if args.exp_name:
        EXP_NAME = args.exp_name
    else:
        EXP_NAME = args.pretrained_model.replace("/", "-")

    # transforms = build_transforms(
    #     args.transforms.copy(),
    #     resize_dims=(224, 224),
    #     sample_frames=args.frames,
    #     random_sample=args.random_sample,
    #     dataset_name=args.dataset,
    # )
    transforms = Transforms(
        args.transforms.copy(),
        resize_dims=(224, 224),
        sample_frames=args.frames,
        random_sample=args.random_sample,
        dataset_name=args.dataset,
        transforms_parameters=args.transforms_parameters,
    )

    dataset_factory = DatasetFactory()
    train_dataset = dataset_factory(
        name=args.dataset,
        root_dir=args.data_path,
        extensions=["pt"],
        transform=transforms,
        split="train",
        specific_classes=args.specific_classes,
        n_samples_per_class=args.n_samples_per_class,
    )
    
    val_dataset = dataset_factory(
        name=args.dataset,
        root_dir=args.data_path,
        extensions=["pt"],
        transform=Transforms(#build_transforms(
            ["normalize"],
            resize_dims=(224, 224),
            sample_frames=args.frames,
            random_sample=False,
            dataset_name=args.dataset,
        ),
        split="test",
        specific_classes=args.specific_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    # callbacks and logger
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

    early_stop_callback = EarlyStopping( #implementar patience como args
        monitor = "val_loss",
        patience=100,
        # min_delta=1e-4,
        verbose=False,
        mode="min",
    ) 

    logger = TensorBoardLogger(save_dir="lightning_logs", name=EXP_NAME)

    # create model
    model = VideoModel(
        args.pretrained_model,
        num_classes=len(train_dataset.classes),
        cache_dir="/mnt/G-SSD/BRACIS/BRACIS-2024/cache",
        lr=args.learning_rate,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        args=args_dict,
    )
    
    if args.finetune:
        print("="*30)
        print("="*15, "Finetuning model", "="*15)
        model.load_pretrained_weights(args.finetune)
        # Adjust the classifier to have the correct number of classes
        classifier_in_features = model.model.classifier.in_features
        model.model.classifier = torch.nn.Linear(classifier_in_features, len(train_dataset.classes))
        print("="*40)

    trainer = L.Trainer(
        log_every_n_steps=11,
        max_epochs=args.max_epochs,
        devices=args.gpus,
        accelerator="gpu",
        strategy="ddp",
        # callbacks=[checkpoint_callback, top5_checkpoint_callback, early_stop_callback],
        callbacks=[checkpoint_callback, top5_checkpoint_callback, early_stop_callback],
        logger=logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
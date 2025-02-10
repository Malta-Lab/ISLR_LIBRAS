import os
import argparse
import torch
import torch.nn as nn
from utils import set_seed, CLASSES2IDX
from dataset_optimized import DatasetFactory, TestDataset
from transforms_optimized import VideoTransforms
from models_optimized import VideoModel
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    top_k_accuracy_score
)
import pandas as pd
import glob
import numpy as np

# Set up argument parsing
parser = argparse.ArgumentParser(description="Evaluate models on specified datasets")
parser.add_argument("-dm", "--dataset_mode", type=str, required=True,
                    choices=["minds_val", "malta_test", "slovo_val", "wlasl_val"],
                    help="Dataset mode for evaluation")
parser.add_argument("--base_dir", type=str, required=True,
                    help="Base directory containing experiment logs")
parser.add_argument("-cuda", "--cuda_device", type=str, default="0",
                    help="CUDA device(s) to use")
parser.add_argument("-bs", "--batch_size", type=int, default=4,
                    help="Batch size for evaluation")
parser.add_argument("-w", "--num_workers", type=int, default=4,
                    help="Number of DataLoader workers")
parser.add_argument("--minds_path", type=str, 
                    default="/mnt/G-SSD/BRACIS/MINDS_tensors_32",
                    help="Path to MINDS dataset")
parser.add_argument("--malta_csv", type=str,
                    default="/mnt/G-SSD/BRACIS/BRACIS-2024/dataset_intersections/matched_labels_with_tensors.csv",
                    help="Path to Malta test CSV")
parser.add_argument("--slovo_path", type=str,
                    default="/mnt/G-SSD/BRACIS/slovo_tensors_32",
                    help="Path to Slovo dataset")
parser.add_argument("--wlasl_path", type=str,
                    default="/mnt/G-SSD/BRACIS/WLASL_tensors_32",
                    help="Path to WLASL dataset")
parser.add_argument("--use_kinetics_norm", action="store_true",
                    help="Use Kinetics normalization instead of dataset-specific")
parser.add_argument("--num_frames", type=int, default=16,
                    help="Number of frames per video clip")

args = parser.parse_args()

# Configure environment
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)
torch.backends.cudnn.benchmark = True

def determine_experiments(base_dir):
    experiments = {}
    for exp_name in os.listdir(base_dir):
        exp_dir = os.path.join(base_dir, exp_name)
        ckpt_path = os.path.join(exp_dir, "checkpoints", "best_model.ckpt")
        
        if os.path.exists(ckpt_path):
            experiments[exp_name] = ckpt_path
        elif os.path.isdir(exp_dir):  # Check for direct checkpoint in experiment dir
            for ckpt in glob.glob(os.path.join(exp_dir, "*.ckpt")):
                experiments[exp_name] = ckpt
    
    if not experiments:
        raise ValueError(f"No valid checkpoints found in {base_dir}")
    
    return experiments, len(experiments) == 1

class ResultAnalyzer:
    def __init__(self, true_labels, pred_labels, class_scores):
        self.true = np.array([CLASSES2IDX[label] if isinstance(label, str) else label 
                             for label in true_labels])
        self.preds = np.array(pred_labels)
        self.scores = np.array(class_scores)
        
        self._validate_inputs()
        
    def _validate_inputs(self):
        if len(self.true) != len(self.preds):
            raise ValueError("Mismatch between true labels and predictions count")
        if self.scores.shape[0] != len(self.true):
            raise ValueError("Score matrix dimension mismatch")
        if self.scores.shape[1] != len(CLASSES2IDX):
            raise ValueError(f"Score matrix columns ({self.scores.shape[1]}) "
                             f"don't match classes ({len(CLASSES2IDX)})")

    def compute_metrics(self):
        return {
            "accuracy": accuracy_score(self.true, self.preds),
            "top5_accuracy": top_k_accuracy_score(self.true, self.scores, k=5),
            "f1_macro": f1_score(self.true, self.preds, average="macro"),
            "precision_macro": precision_score(self.true, self.preds, average="macro"),
            "recall_macro": recall_score(self.true, self.preds, average="macro"),
            "confusion_matrix": confusion_matrix(self.true, self.preds),
        }

def load_dataset():
    common_transforms = {
        "mode": "eval",
        "input_size": (224, 224),
        "num_frames": args.num_frames,
        "use_kinetics_norm": args.use_kinetics_norm,
        "transforms_list": ["normalize"],
    }

    factory = DatasetFactory()
    
    if args.dataset_mode == "minds_val":
        return factory(
            name="minds",
            root_dir=args.minds_path,
            transform=VideoTransforms(**common_transforms),
            split="test"
        )
    elif args.dataset_mode == "malta_test":
        return factory(
            name="test",
            root_dir=args.malta_csv,
            transform=VideoTransforms(**common_transforms)
        )
    elif args.dataset_mode == "slovo_val":
        return factory(
            name="slovo",
            root_dir=args.slovo_path,
            transform=VideoTransforms(**common_transforms),
            split="test"
        )
    elif args.dataset_mode == "wlasl_val":
        return factory(
            name="wlasl",
            root_dir=args.wlasl_path,
            transform=VideoTransforms(**common_transforms),
            split="test"
        )

def eval_collate_fn(batch):
    videos = torch.stack([item[0] for item in batch], dim=0)
    labels = [item[1] for item in batch]
    return videos, labels

def load_model(checkpoint_path):
    model = VideoModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False  # Allow classifier layer mismatch
    )
    
    # Ensure correct classifier dimensions
    if hasattr(model.model, "classifier"):
        if model.model.classifier.out_features != len(CLASSES2IDX):
            print(f"Resetting classifier from {model.model.classifier.out_features} "
                  f"to {len(CLASSES2IDX)} classes")
            in_features = model.model.classifier.in_features
            model.model.classifier = nn.Linear(in_features, len(CLASSES2IDX))
    
    model.eval()
    return model.to(device)

def main():
    exps, single_exp = determine_experiments(args.base_dir)
    all_metrics = []

    for exp_name, ckpt_path in exps.items():
        print(f"\nEvaluating {exp_name}...")
        
        model = load_model(ckpt_path)
        dataset = load_dataset()
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=eval_collate_fn,
            pin_memory=True,
            persistent_workers=args.num_workers > 0
        )

        true_labels, all_scores = [], []
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            for videos, labels in tqdm(loader, desc="Processing batches"):
                outputs = model(videos.to(device, non_blocking=True))
                all_scores.append(outputs.logits.float().cpu().numpy())
                true_labels.extend(labels)

        # Concatenate all results
        all_scores = np.concatenate(all_scores)
        pred_labels = np.argmax(all_scores, axis=1)
        
        # Compute metrics
        analyzer = ResultAnalyzer(true_labels, pred_labels, all_scores)
        metrics = analyzer.compute_metrics()
        
        # Save results
        results = {
            "experiment": exp_name,
            "dataset": args.dataset_mode,
            **metrics,
            "checkpoint": ckpt_path
        }
        
        # Save per-experiment results
        output_dir = args.base_dir if single_exp else os.path.join(args.base_dir, exp_name)
        os.makedirs(output_dir, exist_ok=True)
        
        pd.DataFrame([results]).to_csv(
            os.path.join(output_dir, f"{args.dataset_mode}_metrics.csv"),
            index=False
        )
        
        all_metrics.append(results)

    # Save aggregated results if multiple experiments
    if not single_exp:
        agg_df = pd.DataFrame(all_metrics)
        agg_path = os.path.join(args.base_dir, f"aggregated_{args.dataset_mode}_results.csv")
        agg_df.to_csv(agg_path, index=False)
        print(f"\nSaved aggregated results to {agg_path}")

if __name__ == "__main__":
    main()
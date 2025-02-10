import os
import argparse
import torch
from utils import set_seed
from dataset import *
from transforms import Transforms
from models import VideoModel
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
parser = argparse.ArgumentParser(description="Evaluate models and calculate metrics.")
parser.add_argument("-dm", "--dataset_mode", type=str, default="minds_val",
                    choices=["minds_val", "minds_test","malta_test", "slovo_val", "wlasl_val", "minds_val"],
                    help="Specify the dataset mode for evaluation.")
parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing the experiment logs and models.")
parser.add_argument("-cuda", "--cuda_device", type=str, default="0", help="Specify which CUDA device to use.")
parser.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size for DataLoader.")
parser.add_argument("-w", "--num_workers", type=int, default=8, help="Number of workers for DataLoader.")
args = parser.parse_args()

# Set the CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
set_seed(42)

# Determine if it's a single or multiple experiment setup
def determine_experiments(base_dir):
    if os.path.isdir(base_dir):
        version_dirs = glob.glob(os.path.join(base_dir, "version_*"))
        if len(version_dirs) == 1:
            best_model_path = os.path.join(version_dirs[0], "checkpoints/best_model.ckpt")
            return {os.path.basename(base_dir): best_model_path}, True
        else:
            exps = {}
            for exp_name in os.listdir(base_dir):
                exp_dir = os.path.join(base_dir, exp_name)
                if os.path.isdir(exp_dir):
                    version_dirs = glob.glob(os.path.join(exp_dir, "version_*"))
                    if version_dirs:
                        best_model_path = os.path.join(version_dirs[0], "checkpoints/best_model.ckpt")
                        exps[exp_name] = best_model_path
            return exps, False
    else:
        raise ValueError(f"The directory {base_dir} does not exist or is not a directory.")

EXPS, single_experiment = determine_experiments(args.base_dir)

class ResultAnalyzer:
    def __init__(self, true, preds, scores):
        self.true = np.array([int(label) if isinstance(label, str) else label for label in true])
        self.preds = np.array([int(pred) if isinstance(pred, str) else pred for pred in preds])
        self.scores = np.array(scores)
        # Obtain the number of classes from the scores array
        self.n_classes = self.scores.shape[1]
        # Ensure class_labels includes all possible classes
        self.class_labels = np.arange(self.n_classes)

    def compute_metrics(self):
        acc = accuracy_score(self.true, self.preds)
        f1 = f1_score(self.true, self.preds, average="macro")
        precision = precision_score(self.true, self.preds, average="macro")
        recall = recall_score(self.true, self.preds, average="macro")
        # Use class_labels to ensure the confusion matrix includes all classes
        cm = confusion_matrix(self.true, self.preds, labels=self.class_labels)
        top5_accuracy = top_k_accuracy_score(
            self.true,
            self.scores,
            k=5,
            labels=self.class_labels
        )
        return acc, f1, precision, recall, cm, top5_accuracy

def load_dataset(mode="val", model_name=None):
    if mode == "minds_val":
        return VideoDataset(
            root_dir="/mnt/G-SSD/BRACIS/MINDS_tensors_32",
            extensions=["pt"],
            transform=Transforms(
                transforms_list=["normalize"],
                resize_dims=(224, 224),
                sample_frames=16,
                random_sample=False,
                dataset_name="minds"
            ),
            split="val",
            with_path=True,
        )
    elif mode == "minds_test":
        return VideoDataset(
            root_dir="/mnt/G-SSD/BRACIS/MINDS_tensors_32",
            extensions=["pt"],
            transform=Transforms(
                transforms_list=["normalize"],
                resize_dims=(224, 224),
                sample_frames=16,
                random_sample=False,
                dataset_name="minds"
            ),
            split="test",
            with_path=True,
        )
    
    elif mode == "malta_test":
        # Inside load_dataset() in create_results.py:
    # Load the training dataset to get its class_to_idx
        train_dataset = VideoDataset(
            root_dir="/mnt/G-SSD/BRACIS/MINDS_tensors_32",
            extensions=["pt"],
            transform=Transforms(
                transforms_list=["normalize"],
                resize_dims=(224, 224),
                sample_frames=16,
                random_sample=False,
                dataset_name="minds"
            ),
            split="train",
            with_path=True,
        )
        
        return TestDatasets(
            csv_file="/mnt/G-SSD/BRACIS/BRACIS-2024/dataset_intersections/matched_labels_with_tensors.csv",
            transforms=Transforms(
                transforms_list=["normalize"],
                resize_dims=(224, 224),
                sample_frames=16,
                random_sample=False,
                dataset_name="malta_test"
            ),
            class_to_idx=train_dataset.class_to_idx,
        )
    elif mode == "slovo_val":
        return SlovoDataset(
            dir="/mnt/G-SSD/BRACIS/SLOVO_tensors_32/SLOVO_tensors_32",
            split="test",
            transforms=Transforms(
                transforms_list=["normalize"],
                resize_dims=(224, 224),
                sample_frames=16,
                random_sample=False,
                dataset_name="slovo"
            ),
        )
    elif mode == "wlasl_val":
        return WLASLDataset(
            dir="/mnt/G-SSD/BRACIS/WLASL_tensors_32",
            split="val",
            transforms=Transforms(
                transforms_list=["normalize"],
                resize_dims=(224, 224),
                sample_frames=16,
                random_sample=False,
                dataset_name="wlasl"
            ),
        )
    else:
        raise ValueError(f"Invalid dataset mode: {mode}")

def load_model(model_path):
    model = VideoModel.load_from_checkpoint(model_path)
    model.eval()
    model = model.to(device)
    return model

# Initialize a list to collect metrics DataFrames from all experiments
all_metrics_dfs = []

# Process each experiment (or the single experiment)
for exp_name, model_path in EXPS.items():
    print(f"Processing Experiment: {exp_name}")
    model = load_model(model_path)

    # Load dataset
    dataset = load_dataset(args.dataset_mode)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Initialize dictionaries for tracking results per dictionary
        # With this conditional logic:
    if hasattr(dataset, 'dictionaries'):
        # For TestDatasets
        per_dictionary_results = {dictionary: {"true": [], "preds": [], "scores": []} for dictionary in dataset.dictionaries}
    else:
        # For minds_test/minds_val (VideoDataset)
        per_dictionary_results = {dataset.split: {"true": [], "preds": [], "scores": []}}  # Single ke
    overall_true_labels = []
    overall_predicted_scores = []

    # Evaluate model
    with torch.no_grad():
        for data in tqdm(loader):
            video = data[0].to(device)
            label = data[1].to(device)
            
            # Determine dictionary key
            if hasattr(dataset, 'dictionaries'):
                dictionary = data[2][0]  # Get from batch
            else:
                dictionary = dataset.split  # Use split name
            
            output = model(video)
            scores = output.logits.cpu().numpy()
            # scores[:,9] = -np.inf
            labels = label.cpu().numpy()
            predicted_class = np.argmax(scores, axis=1)
            
            # Append to results
            per_dictionary_results[dictionary]["true"].extend(labels)
            per_dictionary_results[dictionary]["preds"].extend(predicted_class)
            per_dictionary_results[dictionary]["scores"].extend(scores)
            
            # Overall metrics
            overall_true_labels.extend(labels)
            overall_predicted_scores.extend(scores)

    # Analyze results per dictionary
    metrics_list = []
    for dictionary_name, results in per_dictionary_results.items():
        analyzer = ResultAnalyzer(
            true=results["true"],
            preds=results["preds"],
            scores=results["scores"]
        )
        acc, f1, precision, recall, cm, top5_accuracy = analyzer.compute_metrics()

        metrics_data = {
            "model": model.model_name,
            "exp_name": exp_name,
            "dictionary": dictionary_name,
            "acc": acc,
            "top5_acc": top5_accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "cm": cm.tolist()
        }
        metrics_list.append(metrics_data)

    # Overall metrics calculation
    overall_analyzer = ResultAnalyzer(
        true=overall_true_labels,
        preds=np.argmax(overall_predicted_scores, axis=1),
        scores=overall_predicted_scores
    )
    overall_acc, overall_f1, overall_precision, overall_recall, overall_cm, overall_top5_accuracy = overall_analyzer.compute_metrics()

    overall_metrics_data = {
        "model": model.model_name,
        "exp_name": exp_name,
        "dictionary": "Overall",
        "acc": overall_acc,
        "top5_acc": overall_top5_accuracy,
        "f1": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall,
        "cm": overall_cm.tolist()
    }
    metrics_list.append(overall_metrics_data)

    # Save results to CSV inside the experiment folder
    exp_dir = args.base_dir if single_experiment else os.path.join(args.base_dir, exp_name)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(exp_dir, f"{args.dataset_mode}_metrics_results.csv"), index=False)

    # Collect the metrics DataFrame for aggregation
    all_metrics_dfs.append(metrics_df)

# After processing all experiments, compute mean and std if there are multiple experiments
if not single_experiment:
    # Concatenate all metrics DataFrames
    combined_metrics_df = pd.concat(all_metrics_dfs, ignore_index=True)

    # Select the metrics columns to compute mean and std
    metrics_cols = ['acc', 'top5_acc', 'f1', 'precision', 'recall']

    # Group by 'dictionary' to compute metrics per dictionary (including 'Overall')
    grouped = combined_metrics_df.groupby('dictionary')

    # Initialize a DataFrame to hold mean and std results
    mean_std_df = pd.DataFrame()

    # Compute mean and std for each metric
    for metric in metrics_cols:
        mean_series = grouped[metric].mean()
        std_series = grouped[metric].std()

        mean_std_df[f'{metric}_mean'] = mean_series.values
        mean_std_df[f'{metric}_std'] = std_series.values

    # Include the 'dictionary' column
    mean_std_df['dictionary'] = mean_series.index

    # Reorder columns to have 'dictionary' first
    cols = ['dictionary'] + [col for col in mean_std_df.columns if col != 'dictionary']
    mean_std_df = mean_std_df[cols]

    # Save the mean and std metrics to CSV
    mean_std_df.to_csv(os.path.join(args.base_dir, f"{args.dataset_mode}_mean_std_results.csv"), index=False)
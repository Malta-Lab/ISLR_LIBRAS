import os
import argparse
import torch
from utils import set_seed
from dataset import *
from transforms import Transforms
from models import VideoModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import glob

# Set up argument parsing
parser = argparse.ArgumentParser(description="Evaluate models and calculate metrics.")
parser.add_argument("-dm","--dataset_mode", type=str, default="minds_val", 
                    choices=["val", "test", "slovo_val", "wlasl_val", "minds_val"],
                    help="Specify the dataset mode for evaluation. val=")
parser.add_argument("--base_dir", type=str, required=True, 
                    help="Base directory containing the experiment logs and models.")
parser.add_argument("-cuda", "--cuda_device", type=str, default="0", 
                    help="Specify which CUDA device to use.")
parser.add_argument("-bs", "--batch_size", type=int, default=1, 
                    help="Batch size for DataLoader.")
parser.add_argument("-w", "--num_workers", type=int, default=8, 
                    help="Number of workers for DataLoader.")
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
    def __init__(self, true, preds):
        self.true = [int(label) if isinstance(label, str) else label for label in true]
        self.preds = [int(pred) if isinstance(pred, str) else pred for pred in preds]

    def compute_metrics(self):
        accuracy = accuracy_score(self.true, self.preds, normalize=True)
        f1 = f1_score(self.true, self.preds, average="macro")
        precision = precision_score(self.true, self.preds, average="macro")
        recall = recall_score(self.true, self.preds, average="macro")
        cm = confusion_matrix(self.true, self.preds)
        return accuracy, f1, precision, recall, cm

def load_dataset_v3(mode="val", model_name=None):
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
            split="test",
            with_path=True,
        )
    elif mode == "test":
        return TestDatasets(
            csv_file="/mnt/G-SSD/BRACIS/BRACIS-2024/dataset_intersections/matched_labels_with_tensors.csv",
            transforms=Transforms(
                transforms_list=["normalize"],
                resize_dims=(224, 224),
                sample_frames=16,
                random_sample=False,
                dataset_name="test"
            ),
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

all_results = {}

# Process each experiment (or the single experiment)
for exp_name, model_path in EXPS.items():
    print(f"Processing Experiment: {exp_name}")
    model = load_model(model_path)

    # Load dataset
    dataset = load_dataset_v3(args.dataset_mode)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    predictions = []
    true_labels = []

    # Evaluate model
    with torch.no_grad():
        for data in tqdm(loader):
            video, label = data if len(data) >= 2 else (None, None)
            if video is None:
                raise TypeError(f"Unexpected data type returned from dataset. Expected tuple, got {type(data)}.")
            
            video = video.to(device)
            output = model(video)
            pred = torch.argmax(output.logits, dim=1).item()
            
            predictions.append(pred)
            true_labels.append(label.item())

    # Analyze results
    analyzer = ResultAnalyzer(true_labels, predictions)
    acc, f1, precision, recall, cm = analyzer.compute_metrics()

    metrics_data = {
        "model_name": exp_name,
        f"acc_{args.dataset_mode}": acc,
        f"f1_{args.dataset_mode}": f1,
        f"precision_{args.dataset_mode}": precision,
        f"recall_{args.dataset_mode}": recall,
        f"cm_{args.dataset_mode}": cm.tolist()
    }

    # Save results to CSV inside the experiment folder
    exp_dir = args.base_dir if single_experiment else os.path.join(args.base_dir, exp_name)
    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_csv(os.path.join(exp_dir, f"{args.dataset_mode}_metrics_results.csv"), index=False)

    all_results[exp_name] = metrics_data

# If there's more than one experiment, calculate mean and std across all experiments
if not single_experiment:
    mean_std_data = {}
    metrics_keys = [f"acc_{args.dataset_mode}", f"f1_{args.dataset_mode}", f"precision_{args.dataset_mode}", f"recall_{args.dataset_mode}"]

    for key in metrics_keys:
        values = [res[key] for res in all_results.values()]
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        mean_std_data[f"{key}_mean"] = mean
        mean_std_data[f"{key}_std"] = std

    mean_std_df = pd.DataFrame([mean_std_data])
    mean_std_df.to_csv(os.path.join(args.base_dir, f"{args.dataset_mode}_mean_std_results.csv"), index=False)

import os
import argparse
import torch
from utils import set_seed
from dataset_v2 import VideoDataset, TestDatasets
from transforms_v2 import Transforms
from models import VideoModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import glob

# Set up argument parsing
parser = argparse.ArgumentParser(description="Evaluate models and calculate metrics.")
parser.add_argument("--dataset_mode", type=str, default="val", choices=["val", "test"],
                    help="Specify whether to use the validation ('val') or test ('test') dataset.")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(42)

base_dir = "lightning_logs/random_rotation"

EXPS = {}
for exp_name in os.listdir(base_dir):
    exp_dir = os.path.join(base_dir, exp_name)
    if os.path.isdir(exp_dir):
        version_dirs = glob.glob(os.path.join(exp_dir, "version_*"))
        if version_dirs:
            best_model_path = os.path.join(version_dirs[0], "checkpoints/best_model.ckpt")
            EXPS[exp_name] = best_model_path

class ResultAnalyzer:
    def __init__(self, true, preds):
        # Convert labels and predictions to integers if they are strings
        self.true = [int(label) if isinstance(label, str) else label for label in true]
        self.preds = [int(pred) if isinstance(pred, str) else pred for pred in preds]
        self.errors = []

    def compute_metrics(self):
        accuracy = accuracy_score(self.true, self.preds)
        f1 = f1_score(self.true, self.preds, average="macro")
        precision = precision_score(self.true, self.preds, average="macro")
        recall = recall_score(self.true, self.preds, average="macro")
        cm = confusion_matrix(self.true, self.preds)
        return accuracy, f1, precision, recall, cm

    def get_errors(self):
        for i in range(len(self.true)):
            if self.true[i] != self.preds[i]:
                self.errors.append(i)
        return self.errors

def load_dataset(mode="val", dir="../MINDS_tensors_32", model_name=None):
    if mode == "val":
        return VideoDataset(
            root_dir=dir,
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
            csv_file="./dataset_intersections/matched_labels_with_tensors.csv",
            transforms=Transforms(
                transforms_list=["normalize"],
                resize_dims=(224, 224),
                sample_frames=16,
                random_sample=False,
                dataset_name="test"
            ),
        )

def load_model(model_path):
    model = VideoModel.load_from_checkpoint(model_path)
    model.eval()
    model = model.to(device)
    return model

def delete_checkpoints(experiment_path):
    checkpoint_path = os.path.join(experiment_path, "checkpoints")
    try:
        for file in os.listdir(checkpoint_path):
            if file.endswith(".ckpt"):
                os.remove(os.path.join(checkpoint_path, file))
                print(f"Deleted {file} in {checkpoint_path}")
    except Exception as e:
        print(f"Error deleting checkpoints in {checkpoint_path}: {e}")

all_results = {}

# Process each experiment
for k, v in EXPS.items():
    print(f"Experiment: {k}")
    model = load_model(v)
    exp_results = []

    # Load datasets
    dataset = load_dataset(args.dataset_mode, model_name=v)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=32)

    predictions = []
    true_labels = []

    # Evaluate model
    with torch.no_grad():
        for data in tqdm(loader):
            if isinstance(data, list):
                video = data[0]  # tensor of video frames
                label = data[1]  # tensor of label
                # Additional data elements can be handled if needed
            else:
                raise TypeError(f"Unexpected data type returned from dataset. Expected list, got {type(data)}.")
            
            video = video.to(device)
            output = model(video)
            pred = torch.argmax(output.logits, dim=1).item()
            predictions.append(pred)
            true_labels.append(label.item())

    analyzer = ResultAnalyzer(true_labels, predictions)
    acc, f1, precision, recall, cm = analyzer.compute_metrics()

    metrics_data = {
        "model_name": k,
        f"acc_{args.dataset_mode}": acc,
        f"f1_{args.dataset_mode}": f1,
        f"precision_{args.dataset_mode}": precision,
        f"recall_{args.dataset_mode}": recall,
        f"cm_{args.dataset_mode}": cm.tolist()  # Convert confusion matrix to list for CSV compatibility
    }

    # Save results to CSV inside the experiment folder
    exp_dir = os.path.join(base_dir, k)
    metrics_df = pd.DataFrame([metrics_data])
    metrics_df.to_csv(os.path.join(exp_dir, f"{args.dataset_mode}_metrics_results.csv"), index=False)

    all_results.setdefault(k, {}).update(metrics_data)

    # Delete checkpoints after saving metrics
    # delete_checkpoints(exp_dir)

# Calculate mean and std for each metric across all experiments
mean_std_data = {}
metrics_keys = [f"acc_{args.dataset_mode}", f"f1_{args.dataset_mode}", f"precision_{args.dataset_mode}", f"recall_{args.dataset_mode}"]
for key in metrics_keys:
    values = [res[key] for res in all_results.values()]
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values)) ** 0.5
    mean_std_data[f"{key}_mean"] = mean
    mean_std_data[f"{key}_std"] = std

mean_std_df = pd.DataFrame([mean_std_data])
mean_std_df.to_csv(os.path.join(base_dir, f"{args.dataset_mode}_mean_std_results.csv"), index=False)

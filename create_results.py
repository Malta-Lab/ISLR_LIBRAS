import os
import torch
from utils import set_seed
from dataset import VideoDataset, TestDatasets
from transforms import build_transforms
from models import VideoModel
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import pandas as pd


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

EXPS = {
    exp_name: f"lightning_logs/{exp_name}/version_0/checkpoints/best_model.ckpt"
    for exp_name in os.listdir("lightning_logs")
    if os.path.isdir(f"lightning_logs/{exp_name}")
}


class ResultAnalyzer:
    def __init__(self, true, preds):
        self.true = [i[1] for i in true]
        self.paths = [i[0] for i in true]
        self.preds = preds
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
                self.errors.append(self.paths[i])
        return self.errors


def count_videos_by_class():
    for root, _, files in os.walk("../MINDS_tensors"):
        print(f"{root} has {len(files)} tensors")


def load_dataset(
    mode="val",
    dir="../MINDS_tensors_all_frames",
    model_name=None,
):
    if mode == "val":
        return VideoDataset(
            root_dir=dir,
            extensions=["pt"],
            transform=build_transforms(
                ["normalize"],
                resize_dims=(224, 224),
                sample_frames=16 if "vivit" not in model_name else 32,
                random_sample=False,
            ),
            split="test",
            with_path=True,
        )
    else:
        return TestDatasets(
            "./dataset_intersections/common_labels.csv",
            transforms=build_transforms(
                (["normalize"]),
                resize_dims=(224, 224),
                sample_frames=16 if "vivit" not in model_name else 32,
                random_sample=False,
            ),
        )


def load_model(model_path):
    model = VideoModel.load_from_checkpoint(model_path)
    model.eval()
    model = model.to(device)
    return model


# run every experiment for all the three datasets and get the values of the metrics
all_results = {e: [] for e in EXPS.keys()}

for k, v in EXPS.items():
    print(f"Experiment: {k}")
    model = load_model(v)
    exp_results = []

    val_all_frames = load_dataset("val", "../MINDS_tensors_all_frames", model_name=v)
    val_32_frames = load_dataset("val", "../MINDS_tensors_32", model_name=v)
    test_dataset = load_dataset("test", model_name=v)
    loader_val_all_frames = torch.utils.data.DataLoader(
        val_all_frames,
        batch_size=1,
        shuffle=False,
        num_workers=32,
    )
    loader_val_32_frames = torch.utils.data.DataLoader(
        val_32_frames,
        batch_size=1,
        shuffle=False,
        num_workers=32,
    )
    loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    loaders = [loader_val_all_frames, loader_val_32_frames, loader_test]

    with torch.no_grad():
        for loader in loaders:
            dataset_results = []
            for video, label, origin, path in tqdm(loader):
                video = video.to(device)
                output = model(video)
                pred = torch.argmax(output.logits, dim=1)
                dataset_results.append(pred.item())
            exp_results.append(dataset_results)
    all_results[k] = exp_results


labels_all_frames = val_all_frames.samples
labels_test = test_dataset.df["label"].values
dictionary_test = test_dataset.df["dictionary"].values
paths_test = test_dataset.df["path"].values


final_csv = {}
for k, v in all_results.items():
    print(k, v)
    ra_32 = ResultAnalyzer(labels_all_frames, v[0])
    ra_all = ResultAnalyzer(labels_all_frames, v[1])

    acc_32, f1_32, precision_32, recall_32, cm_32 = ra_32.compute_metrics()
    acc_all, f1_all, precision_all, recall_all, cm_all = ra_all.compute_metrics()
    errors_32 = ra_32.get_errors()
    errors_all = ra_all.get_errors()

    final_csv[k] = {
        "model_name": k,
        "acc_32": acc_32,
        "f1_32": f1_32,
        "precision_32": precision_32,
        "recall_32": recall_32,
        "cm_32": cm_32,
        "errors_32": errors_32,
        "acc_all": acc_all,
        "f1_all": f1_all,
        "precision_all": precision_all,
        "recall_all": recall_all,
        "cm_all": cm_all,
        "errors_all": errors_all,
    }

# save dataframe to csv
df = pd.DataFrame.from_dict(final_csv, orient="index")
df.to_csv("results/val_results.csv")

true_test = [t.split("_")[0] for t in labels_test]
true_test = [
    t.replace("á", "a")
    .replace("ã", "a")
    .replace("é", "e")
    .replace("í", "i")
    .replace("ó", "o")
    .replace("ú", "u")
    .replace("ç", "c")
    for t in true_test
]
true_test = [val_all_frames.class_to_idx[t] for t in true_test]

test_dataset.df["number_label"] = true_test

true_test = [(path, t) for path, t in zip(paths_test, true_test)]

final_csv = {}

for k, v in all_results.items():
    print(k, v)
    ra_test = ResultAnalyzer(true_test, v[2])

    acc, f1, precision, recall, cm = ra_test.compute_metrics()
    errors = ra_test.get_errors()

    final_csv[k] = {
        "model_name": k,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "cm": cm,
        "errors": errors,
    }

# save dataframe to csv
df = pd.DataFrame.from_dict(final_csv, orient="index")
df.to_csv("results/results_test.csv")


for k, v in all_results.items():
    test_dataset.df[k] = v[2]

test_dataset.df.to_csv("results/test_results_model_by_column.csv")

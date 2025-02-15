import os
import ast
import torchvision
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import Compose
torchvision.disable_beta_transforms_warning()
from torchvision.io import read_video
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import CLASSES2IDX

class DatasetFactory:
    def __call__(
        self,
        name="minds",
        root_dir=None,
        transform=None,
        extensions=["pt"],
        split="train",
        seed=42,
        specific_classes=None,
        n_samples_per_class=None,
        with_path=False,
    ):

        if name == "minds" and split == "train":
            return VideoDataset(
                root_dir,
                transform,
                extensions,
                split,
                seed,
                specific_classes,
                n_samples_per_class,
                with_path,
            )
        elif name == "minds" and split == "test":
            return VideoDataset(
                root_dir,
                transform,
                extensions,
                split,
                seed,
                specific_classes,
                n_samples_per_class,
                with_path,
            )
        elif name == "test":
            return TestDatasets(root_dir, transform)
        elif name == "slovo":
            return SlovoDataset(root_dir, split, transform)
        elif name == "wlasl":
            return WLASLDataset(root_dir, split, transform)
        else:
            raise ValueError(f"Invalid dataset name: {name}")


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        extensions=["pt"],
        split="train",
        seed=42,
        specific_classes=None,
        n_samples_per_class=None,
        with_path=False,
    ):
        """
        Args:
            root_dir (str): Directory with all the video files organized in subfolders per class.
            transform (callable, optional): Optional transform to be applied on a sample.
            extensions (list): List of allowed video file extensions.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = extensions
        self.seed = seed
        self.split = split
        self.with_path = with_path
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)

        self.samples = self._make_dataset(
            self.root_dir, self.class_to_idx, self.extensions
        )

        self.samples = self.__get_split_by_sign(self.split)

        if n_samples_per_class:
            self.samples = self.__set_number_of_videos_per_class(n_samples_per_class)

        if specific_classes:
            self.samples = [
                sample
                for sample in self.samples
                if self.classes[sample[1]] in specific_classes
            ]

            self.classes = specific_classes
            self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

    def _find_classes(self, dir):
        """Finds the class folders in a dataset."""
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx, extensions):
        """Creates the dataset by scanning for video files."""
        instances = []
        dir = os.path.expanduser(dir)
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if any(fname.lower().endswith(ext) for ext in extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_index)
                        instances.append(item)
        return instances

    def __get_split_by_sign(self, split):
        train_samples = []
        test_samples = []

        sign_groups = {}
        for path, class_index in self.samples:
            sign = self.classes[class_index]
            if sign not in sign_groups:
                sign_groups[sign] = []
            sign_groups[sign].append((path, class_index))

        for sign, group_samples in sign_groups.items():
            train, test = train_test_split(
                group_samples, test_size=0.25, random_state=self.seed
            )

            train_samples.extend(train)
            test_samples.extend(test)

        if split == "train":
            print(f"Train size: {len(train_samples)}")
            return train_samples
        elif split == "test":
            print(f"Test size: {len(test_samples)}")
            return test_samples
        else:
            raise ValueError(f"Invalid split: {split}")

    def __set_number_of_videos_per_class(self, n_samples_per_class):
        samples = []

        # order the samples, to always ensure the same order
        self.samples = sorted(self.samples, key=lambda x: x[0])
        for class_index in range(len(self.classes)):
            samples.extend(
                [sample for sample in self.samples if sample[1] == class_index][
                    0:n_samples_per_class
                ]
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        video = torch.load(path)

        if self.transform is not None:
            video = self.transform(video)

        return video, target, path if self.with_path else video, target

class TestDatasets(Dataset):
    def __init__(self, csv_file, transforms=None, class_to_idx=None):
        self.df = pd.read_csv(csv_file)
        self.transforms = transforms if transforms is not None else Compose([])
        self.df["label"] = self.df["label"].str.lower()
        
        # Use provided class mapping or fall back to global CLASSES2IDX
        self.class_to_idx = class_to_idx if class_to_idx is not None else CLASSES2IDX
        
        # Filter out any labels not in our class mapping
        valid_labels = [k.lower() for k in self.class_to_idx.keys()]
        self.df = self.df[self.df["label"].isin(valid_labels)]
        
        # Create reverse mapping and get class list
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        self.dictionaries = sorted(self.df["dictionary"].unique())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = ast.literal_eval(row["tensor_path"])[0]
        video = torch.load(video_path).float()
        
        if self.transforms:
            video = self.transforms(video)
            
        label_idx = self.class_to_idx[row["label"].lower()]
        return video, label_idx, row["dictionary"], video_path


class SlovoDataset(Dataset):
    def __init__(self, dir, split="train", transforms=None):
        self.dir = Path(dir)
        self.split = split
        self.transforms = transforms
        self.annotations = pd.read_csv(self.dir / f"annotations.tsv", sep="\t")

        if self.split == "train":
            self.annotations = self.annotations[self.annotations["train"] == True]
        else:
            self.annotations = self.annotations[self.annotations["train"] == False]

        self.labels2idx = self.__getlabels()
        self.classes = list(self.labels2idx.keys())

    def __len__(self):
        return len(self.annotations)

    def __getlabels(self):
        labels = list(self.annotations["text"].unique())
        labels.sort()
        return {label: i for i, label in enumerate(labels)}

    def __getitem__(self, idx):
        instance = self.annotations.iloc[idx]
        video = torch.load(self.dir / self.split / f"{instance['attachment_id']}.pt")

        if self.transforms:
            video = self.transforms(video)

        return video, self.labels2idx[instance["text"]]


class WLASLDataset(Dataset):
    def __init__(self, dir, split="train", transforms=None):
        self.dir = Path(dir)  
        self.split = split
        self.transforms = transforms
        
        self.labels2idx = self.__load_labels(self.dir / "wlasl_class_list.txt")
        self.idx2labels = {v: k for k, v in self.labels2idx.items()}

        self.annotations = pd.read_json(self.dir / "nslt_2000.json").T
        self.annotations["id"] = self.annotations.index
        self.annotations.reset_index(drop=True, inplace=True)

        self.annotations['id'] = self.annotations['id'].apply(lambda x: f'{int(x):05}')
        
        self.annotations["label"] = self.annotations["action"].apply(lambda x: self.labels2idx.get(x[0], "Unknown"))

        # self.missing = self.__get_missing()

        # self.annotations = self.annotations[~self.annotations['id'].isin(self.missing)]
        
        if self.split == "train":
            self.annotations = self.annotations[self.annotations["subset"] != "test"]
            print(f"Train size: {len(self.annotations)}")
        else:
            self.annotations = self.annotations[self.annotations["subset"] == "test"]
            print(f"Test size: {len(self.annotations)}")

        self.classes = list(self.labels2idx.values())
        

    def __load_labels(self, class_list_path):
        """
        Load the label-to-index mapping from the wlasl_class_list.txt file.
        """
        labels2idx = {}
        with open(class_list_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                label_id = int(parts[0])
                label = parts[1]
                labels2idx[label_id] = label
                
        return labels2idx
    
    # def __get_missing(self):
    #     """
    #     Read the missing instances from the missing.txt file and ensure they are zero-padded.
    #     """
    #     missing = []
    #     missing_file = self.dir / "new_missing.txt"
    #     if missing_file.exists():
    #         with open(missing_file, 'r') as file:
    #             missing = [f'{int(line.strip()):05}' for line in file if line.strip()]
    #     return missing

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        
        row = self.annotations.iloc[idx]
        video_id = row["id"]
        
        label = row["label"]
        
        label_idx = self.idx2labels[label]
        
        video_path = self.dir / "videos" / f"{video_id}.pt"
        
        try:
            video = torch.load(video_path)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Video file {video_path} not found.")
        
        if self.transforms:
            video = self.transforms(video)

        return video, label_idx 
import os
import ast
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple, Union
from utils import CLASSES2IDX

class DatasetFactory:
    def __call__(
        self,
        name: str = "minds",
        root_dir: Optional[str] = None,
        transform: Optional[callable] = None,
        split: str = "train",
        seed: int = 42,
        specific_classes: Optional[List[str]] = None,
        **kwargs
    ) -> Dataset:
        
        if name == "minds":
            return VideoDataset(
                root_dir=root_dir,
                transform=transform,
                split=split,
                seed=seed,
                specific_classes=specific_classes
            )
        elif name == "slovo":
            return SlovoDataset(
                root_dir=root_dir,
                split=split,
                transform=transform
            )
        elif name == "wlasl":
            return WLASLDataset(
                root_dir=root_dir,
                split=split,
                transform=transform
            )
        elif name == "test":
            return TestDataset(
                csv_file=root_dir,
                transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {name}")

class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[callable] = None,
        split: str = "train",
        seed: int = 42,
        specific_classes: Optional[List[str]] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.seed = seed
        
        # Metadata collection
        self.samples, self.classes = self._collect_metadata()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Class-balanced split
        self._train_test_split()
        
        # Class filtering
        if specific_classes:
            self._filter_classes(specific_classes)

    def _collect_metadata(self) -> Tuple[List[Tuple[str, str]], List[str]]:
        classes = sorted([d.name for d in os.scandir(self.root_dir) if d.is_dir()])
        samples = []
        for cls in classes:
            cls_dir = self.root_dir / cls
            files = list(cls_dir.glob("*.pt"))
            samples.extend((str(f), cls) for f in files)
        return samples, classes

    def _train_test_split(self):
        self.train_samples = []
        self.test_samples = []
        for cls in self.classes:
            cls_samples = [s for s in self.samples if s[1] == cls]
            train, test = train_test_split(
                cls_samples,
                test_size=0.25,
                random_state=self.seed
            )
            self.train_samples.extend(train)
            self.test_samples.extend(test)
            
        self.samples = self.train_samples if self.split == "train" else self.test_samples

    def _filter_classes(self, classes: List[str]):
        valid = {c.lower() for c in classes}
        original_count = len(self.samples)
        
        self.samples = [
            (p, c) for p, c in self.samples
            if c.lower() in valid
        ]
        self.classes = [c for c in self.classes if c.lower() in valid]
        
        if not self.classes:
            invalid_classes = [c for c in classes if c.lower() not in valid]
            raise ValueError(
                f"No valid classes found. Invalid classes: {invalid_classes}\n"
                f"Valid classes: {list(self.class_to_idx.keys())}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, cls = self.samples[idx]
        video = torch.load(path, map_location='cpu')
        if self.transform:
            video = self.transform(video)
        return video, self.class_to_idx[cls]

class SlovoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[callable] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load annotations
        self.annotations = pd.read_csv(
            self.root_dir / "annotations.tsv",
            sep="\t"
        )
        self.annotations = self.annotations[self.annotations["train"] == (split == "train")]
        
        # Label mapping
        self.classes = sorted(self.annotations["text"].unique())
        
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.annotations.iloc[idx]
        video_path = self.root_dir / self.split / f"{row['attachment_id']}.pt"
        video = torch.load(video_path, map_location='cpu')
        if self.transform:
            video = self.transform(video)
        return video, self.class_to_idx[row["text"]]

class WLASLDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[callable] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Load class list
        self.classes = self._load_classes()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load and filter annotations
        with open(self.root_dir / "nslt_2000.json") as f:
            all_annots = json.load(f)
            
        self.annotations = []
        for video_id, metadata in all_annots.items():
            # Remap splits: val->train, test->val
            if split == "train" and metadata["subset"] in ["train", "val"]:
                self.annotations.append((video_id, metadata))
            elif split == "test" and metadata["subset"] == "test":
                self.annotations.append((video_id, metadata))

    def _load_classes(self) -> List[str]:
        """Load from wlasl_class_list.txt with index validation"""
        classes = []
        with open(self.root_dir / "wlasl_class_list.txt") as f:
            for line in f:
                idx, cls = line.strip().split("\t")
                assert int(idx) == len(classes), "Class index mismatch"
                classes.append(cls)
        return classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_id, metadata = self.annotations[idx]
        class_idx = metadata["action"][2]  # Third element is class
        
        try:
            # Load video tensor
            video_path = self.root_dir / "videos" / f"{video_id}.pt"
            video = torch.load(video_path, map_location='cpu')
            
            # Validate class index
            if class_idx >= len(self.classes):
                raise ValueError(f"Class index {class_idx} out of range")
                
            if self.transform:
                video = self.transform(video)
                
            return video, class_idx
            
        except (FileNotFoundError, ValueError) as e:
            # Skip missing files/invalid classes
            print(f"Skipping video {video_id}: {e}")
            return self[(idx + 1) % len(self)]

class TestDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        transform: Optional[callable] = None
    ):
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.classes = list(CLASSES2IDX.keys())
        self.class_to_idx = CLASSES2IDX
        
        # Preprocess paths
        self.samples = []
        for _, row in self.df.iterrows():
            try:
                path = ast.literal_eval(row["tensor_path"])[0]
                label = row["label"].lower()
                if label in self.class_to_idx:
                    self.samples.append((path, label))
            except:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        video = torch.load(path, map_location='cpu')
        if self.transform:
            video = self.transform(video)
        return video, self.class_to_idx[label]
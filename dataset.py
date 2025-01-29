import os
import ast
import torchvision
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict

torchvision.disable_beta_transforms_warning()
from torchvision.io import read_video
from torchvision.transforms import Compose
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
        if name == "minds":
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
    _split_cache = {}

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
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = extensions
        self.seed = seed
        self.split = split
        self.with_path = with_path
        
        self.classes = list(CLASSES2IDX.keys())
        self.class_to_idx = CLASSES2IDX
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Cache key based on dataset configuration
        cache_key = (root_dir, seed)
        
        # Compute splits only once per configuration
        if cache_key not in VideoDataset._split_cache:
            all_samples = self._make_dataset(root_dir, self.class_to_idx, extensions)
            VideoDataset._split_cache[cache_key] = self.__get_split_by_sign(all_samples)
            
            # Print split verification only once
            splits = VideoDataset._split_cache[cache_key]
            total = len(all_samples)
            print(f"\nDataset splits (Total: {total} samples):")
            print(f"Train: {len(splits['train'])} ({len(splits['train'])/total:.1%})")
            print(f"Val: {len(splits['val'])} ({len(splits['val'])/total:.1%})")
            print(f"Test: {len(splits['test'])} ({len(splits['test'])/total:.1%})\n")

        # Get cached splits
        self.samples = VideoDataset._split_cache[cache_key][split]

        # Apply sampling if requested
        if n_samples_per_class:
            self.samples = self.__set_number_of_videos_per_class(n_samples_per_class)

        # Filter classes if specified
        if specific_classes:
            valid_classes = [cls for cls in specific_classes if cls in self.class_to_idx]
            if len(valid_classes) != len(specific_classes):
                raise ValueError("Some classes in specific_classes are not in CLASSES2IDX.")
            
            original_indices = [self.class_to_idx[cls] for cls in valid_classes]
            new_indices = {original: new for new, original in enumerate(original_indices)}
            
            filtered_samples = []
            for path, original_label in self.samples:
                if original_label in original_indices:
                    new_label = new_indices[original_label]
                    filtered_samples.append((path, new_label))
            
            self.samples = filtered_samples
            self.classes = valid_classes
            self.class_to_idx = {cls: idx for idx, cls in enumerate(valid_classes)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(valid_classes)}

        print(f"Final {split} set samples: {len(self.samples)}")

    def _make_dataset(self, dir, class_to_idx, extensions): # this function is used to create the dataset
        instances = []
        dir = os.path.expanduser(dir)
        dir_entries = {}

        for entry in os.scandir(dir):
            if entry.is_dir():
                dir_name_lower = entry.name.lower()
                dir_entries[dir_name_lower] = entry

        for target_class, class_index in class_to_idx.items():
            target_class_lower = target_class.lower()
            if target_class_lower not in dir_entries:
                print(f"Warning: Directory for class '{target_class}' not found.")
                continue
            
            entry = dir_entries[target_class_lower]
            target_dir = entry.path
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if any(fname.lower().endswith(ext) for ext in extensions):
                        path = os.path.join(root, fname)
                        instances.append((path, class_index))
        return instances

    def __get_split_by_sign(self, all_samples): # setting splits to 40/40/20, ensuring each sign is present in all splits
        split_cache = {
            'train': [],
            'val': [],
            'test': []
        }

        sign_groups = defaultdict(list)
        for path, class_index in all_samples:
            sign = self.idx_to_class[class_index]
            sign_groups[sign].append((path, class_index))

        for sign, group_samples in sign_groups.items():
            # First split: 80% (train+val) vs 20% test
            train_val, test = train_test_split(
                group_samples, 
                test_size=0.20,
                random_state=self.seed
            )
            
            # Second split: 50/50 of remaining 80%
            train, val = train_test_split(
                train_val, 
                test_size=0.50,
                random_state=self.seed
            )
            
            split_cache['train'].extend(train)
            split_cache['val'].extend(val)
            split_cache['test'].extend(test)

        return split_cache

    def __set_number_of_videos_per_class(self, n_samples_per_class): # setting the number of videos per class
        samples = []
        sorted_samples = sorted(self.samples, key=lambda x: x[0])
        for class_index in range(len(self.classes)):
            class_samples = [s for s in sorted_samples if s[1] == class_index]
            samples.extend(class_samples[:n_samples_per_class])
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index): # this function is used to get the item from the dataset
        path, target = self.samples[index]
        video = torch.load(path)

        if self.transform is not None:
            video = self.transform(video)

        return (video, target, path) if self.with_path else (video, target)

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

        self.annotations = pd.read_json(self.dir / "nslt_1000.json").T
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
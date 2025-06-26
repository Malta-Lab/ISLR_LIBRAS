# ISLR_LIBRAS ✨

**ISLR_LIBRAS** is a toolkit and dataset repository from Malta Lab for benchmarking sign-language recognition models. It provides:

- ✔️ A curated Brazilian Sign Language dataset (MALTA-LIBRAS) and its intersections with other open datasets  
- ✔️ Scripts to download, preprocess, and convert videos into tensors  
- ✔️ End-to-end training and evaluation pipelines using PyTorch Lightning  
- ✔️ Utilities to aggregate results and reproduce experiments from our paper  

---

## 📑 Table of Contents

- [🗂️ Data Downloading](#data-downloading)  
- [⚙️ Dataset Preparation](#dataset-preparation)  
- [🛠️ Installation](#installation)  
- [🚀 Usage](#usage)  
- [🧰 Scripts Overview](#scripts-overview)  
- [📁 Directory Structure](#directory-structure)  
- [📊 Evaluation & Results](#evaluation--results)  
- [🤝 Contributing](#contributing)  
- [📄 License](#license)  
- [📫 Contact](#contact)  

---

## 🗂️ Data Downloading

Due to data policies, we can’t host the videos directly, but all sources are public! 🙌

**Sources for each dictionary in MALTA-LIBRAS** (web-scraping via 'videos_download/download_videos.py'):  
- **INES V2** 📘: http://www.acessibilidadebrasil.org.br/libras/  
- **INES V3** 📗: http://www.acessibilidadebrasil.org.br/libras_3/  
- **Corpus Libras** (UFSC) 📙: https://corpuslibras.ufsc.br/  
- **SignBank** (UFSC_V2) 📓: https://signbank.libras.ufsc.br/pt  
- **Spread the Sign** 🌐: https://www.spreadthesign.com/pt.br/search/  
- **V-LIBRASIL** (UFPE) 🎥: https://libras.cin.ufpe.br/  
- **USP** 🏛️: https://edisciplinas.usp.br/mod/glossary/view.php?id=197645  
- **UFV** 🎓: https://sistemas.cead.ufv.br/capes/dicionario/  
- **YouTube** ▶️: a curated SharePoint link is provided in the repo for direct download.  

**Other datasets**:  
- **WLASL** (American SL) 🇺🇸: download via [Kaggle](https://www.kaggle.com/datasets/utsavk02/wlasl-complete)  
- **SLOVO** (Czech SL) 🇨🇿: clone from https://github.com/hukenovs/slovo.git  

Annotations:  
- 💡 Full MALTA-LIBRAS glossary: 'dataset_intersections/glossary.csv'  
- 💡 Experiment subset: 'dataset_intersections/matched_labels_with_tensors.csv'  

---

## ⚙️ Dataset Preparation

1. **Build tensors for public benchmarks**  
   bash
   python build_tensor_dataset.py \
     --wl dataset_path/WLASL \
     --slovo dataset_path/SLOVO \
     --minds dataset_path/MINDS \
     --output_dir tensors/

2. **Build tensors for MALTA_LIBRAS**  
   bash
   python vuild_test_dataset_tensors.py \
   --annotations dataset_intersections/glossary.csv \
   --videos_dir path/to/downloaded_videos \
   --output_dir tensors/malta_libras

## 🛠️ Installation
### Clone the repo

git clone https://github.com/Malta-Lab/ISLR_LIBRAS.git
cd ISLR_LIBRAS

### (Optional) Create a virtual environment

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

### (Optional) Extras for notebooks

pip install jupyterlab

## 🚀 Usage
### 🏋️‍♂️ Training
Launch training with:

python train.py \
  --data_dir tensors/ \
  --dataset MALTA_LIBRAS \
  --model resnet3d \
  --epochs 50 \
  --batch_size 16 \
  --gpus 1 \
  --seed 42 \
  --output_dir lightning_logs/

### 📈 Results Aggregation

After training, generate results:

python create_results.py \
  --logs_dir lightning_logs/ \
  --output_csv results/summary.csv

## 🧰 Scripts Overview

build_tensor_dataset.py — prepare tensors for WLASL, SLOVO, MINDS

build_test_dataset_tensors.py — prepare tensors for MALTA-LIBRAS

train.py — model training pipeline (PyTorch Lightning)

create_results.py — aggregate .csv outputs into summary tables

delete_ckpt.py — clean up old checkpoints

metrics_evaluation.ipynb — analyze experiment results

dataset_intersections/ — CSVs of overlapping labels

videos_download/ — scripts to fetch videos

## 📁 Directory Structure

ISLR_LIBRAS/
├── build_tensor_dataset.py
├── build_test_dataset_tensors.py
├── create_results.py
├── train.py
├── dataset.py
├── models.py
├── transforms.py
├── utils.py
├── requirements.txt
├── seeds.txt
├── metrics_evaluation.ipynb
├── dataset_intersections/
├── videos_download/
└── lightning_logs/

## 📊 Evaluation & Results

Use metrics_evaluation.ipynb to visualize per-class accuracy, confusion matrices, and learning curves.

Checkpoints, logs, and CSVs are stored under lightning_logs/<experiment_name>/.

## 📄 License
This repo is under MIT license.

## 📫 Contact

Malta Lab – https://github.com/Malta-Lab

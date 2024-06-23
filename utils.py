import os
import random
import torch
import numpy as np
from torchvision.io import read_video
from torchvision.transforms.v2 import Resize, Normalize, UniformTemporalSubsample


def read_single_video(video_as_mp4, device):
    frames = read_video(video_as_mp4)[0].float()
    frames = frames.permute(3, 0, 1, 2)
    frames = frames.permute(1, 0, 2, 3)
    frames = Resize((224, 224))(frames)
    frames = Normalize((118.4939, 118.4997, 118.5007), (57.2457, 57.2454, 57.2461))(
        frames
    )
    frames = UniformTemporalSubsample(16)(frames)
    frames = frames.unsqueeze(0)
    frames = frames.to(device)
    return frames
    
def create_labels2idx(dataset):
    label2idx = []

    for (video, label) in dataset._labeled_videos:
        label_text = video.split('/')[2]
        label2idx.append((label_text, label['label']))
        
    return label2idx

def set_seed(seed=42):

    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
# class Results():
#     def __init__(self):
#         self.true_labels = []
#         self.pred_labels = []
#         self.wrong_paths = []
#         self.accuracyTm = Accuracy(task='multiclass', num_classes=20).to(device)
#         self.f1Tm = F1Score(task='multiclass', num_classes=20).to(device)
#         self.cmTm = ConfusionMatrix(task='multiclass', num_classes=20).to(device)
#         self.precisionTm = Precision(task='multiclass', num_classes=20).to(device)
#         self.recallTm = Recall(task='multiclass', num_classes=20).to(device)
        
#     def __append(self, true, pred):
#         self.true_labels.append(true)
#         self.pred_labels.append(pred)
        
        
#     def __call__(self, true, pred):
#         self.__append(true, pred)
#         self.accuracyTm(pred, true)
#         self.f1Tm(pred, true)
#         self.cmTm(pred, true)
#         self.precisionTm(pred, true)
#         self.recallTm(pred, true)
        
#     def add_to_wrong_paths(self, path):
#         self.wrong_paths.append(path)
        
#     def __compute_accuracy(self):
#         accuracy = sum([i for i, j in zip(self.true_labels, self.pred_labels) if i == j]) / len(self.true_labels)
#         self.accuracyTm.compute()
#         return accuracy
    
#     def __compute_confusion_matrix(self):
#         confusion_matrix = torch.zeros(20, 20)
#         for t, p in zip(self.true_labels, self.pred_labels):
#             confusion_matrix[t, p] += 1
#         self.cmTm.compute()
#         return confusion_matrix

#     def __compute_cm_metrics(self):
#         precision = self.cm.diag() / self.cm.sum(1)
#         recall = self.cm.diag() / self.cm.sum(0)
#         f1 = 2 * precision * recall / (precision + recall) 
#         self.precisionTm.compute(), self.recallTm.compute(), self.f1Tm.compute()
#         return precision, recall, f1
    
#     def compute_metrics(self):
#         self.accuracy = self.__compute_accuracy()
#         self.cm = self.__compute_confusion_matrix()
#         self.precision, self.recall, self.f1 = self.__compute_cm_metrics()
        
#     def plot_confusion_matrix(self):
#         df_cm = pd.DataFrame(self.cm.numpy(), index = [i for i in range(20)],
#                     columns = [i for i in range(20)])
#         plt.figure(figsize = (10,7))
#         sns.heatmap(df_cm, annot=True)
#         plt.show()
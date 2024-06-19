import lightning as L
from transformers import AutoModelForVideoClassification, AutoImageProcessor
import torch.optim as optim
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

class VideoModel(L.LightningModule):
    def __init__(self, model_name, 
                 num_classes, 
                 cache_dir=None,
                 lr = 1e-3,
                 optimizer='adamw',
                 scheduler=None,
                 args=None):
        
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.cache_dir = cache_dir
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        
        # Load the model
        
        self.model = AutoModelForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )
        		
		# metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes, average='macro')
        self.f1_train = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        # self.cm_train = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes, average='macro')
        self.f1_val = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        # self.cm_val = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        
        self.save_hyperparameters()
  
          
    def forward(self, x):
        return self.model(x)
        
        
    def training_step(self, batch, batch_idx):
        x,y = batch[0], batch[1]
        outputs = self.model(x, labels=y)
        loss = outputs.loss
        
        self.train_acc(outputs.logits, y)
        self.f1_train(outputs.logits, y)
        # self.cm_train(outputs.logits, y)
        
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', self.train_acc, sync_dist=True, on_step=False, on_epoch=True)
        self.log('f1_train', self.f1_train, sync_dist=True, on_step=False, on_epoch=True)
        # self.log('cm_train', self.cm_train, sync_dist=True, on_step=False, on_epoch=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x,y = batch[0], batch[1]
        outputs = self.model(x, labels=y)
        loss = outputs.loss
        
        self.val_acc(outputs.logits, y)
        self.f1_val(outputs.logits, y)
        # self.cm_val(outputs.logits, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_acc', self.val_acc, sync_dist=True, on_step=False, on_epoch=True)
        self.log('f1_val', self.f1_val, sync_dist=True, on_step=False, on_epoch=True)
        # self.log('cm_val', self.cm_val, sync_dist=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        if self.optimizer == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError('Invalid optimizer')

        if self.scheduler:
            if self.scheduler == 'plateau':
                sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            elif self.scheduler == 'step':
                sched = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

            return {'optimizer': optimizer, 'lr_scheduler': sched, 'monitor': 'val_loss' }

        return optimizer
    
    def on_train_epoch_end(self):
        self.train_acc.compute()
        
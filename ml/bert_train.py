import os
import time
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase, 
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    AdamW,
    get_scheduler
)

from sklearn.metrics import log_loss, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

VER = 'bert-try' # version of exp

os.environ['WANDB_API_KEY'] = 'no please'
os.environ['WANDB_PROJECT'] = 'FIC RuBERT'
os.environ['WANDB_NOTES'] = f'FIC RuBERT Training VER-{VER}'
os.environ['WANDB_NAME'] = f'ft-rubert-fic-ver-{VER}'

@dataclass
class Config:
    output_dir: str = 'output'
    checkpoint: str = 'Tochka-AI/ruRoPEBert-e5-base-2k'
    hdim: int = 768
    num_labels: int = 2
    max_length: int = 2048
    optim_type: str = 'adamw_torch'
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    per_device_eval_batch_size: int = 4
    n_epochs: int = 5
    lr: float = 5e-5
    warmup_steps: int = 60
    
config = Config()

training_args = TrainingArguments(
    output_dir=f'output-{VER}',
    overwrite_output_dir=True,
    report_to='wandb',
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    logging_steps=1,
    eval_strategy='epoch',
    save_strategy='steps',
    save_steps=700,
    optim=config.optim_type,
    learning_rate=config.lr,
    warmup_steps=config.warmup_steps,
    lr_scheduler_type='cosine',  # 'cosine' or 'linear' or 'constant' (default is 'linear')
)

tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(
    config.checkpoint,
    num_labels=config.num_labels,
    device_map='cuda',
)

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.1),
    torch.nn.Linear(config.hdim, config.hdim // 2),
    torch.nn.Dropout(0.1),
    torch.nn.GELU(),
    torch.nn.Linear(config.hdim // 2, config.num_labels),
)

train = pd.read_json('data/client_dataset.json')
train = train.fillna('нет информации')

def remove_duplicate_blocks(text: str) -> str:
    lines = text.split('\n')

    unique_blocks = set()
    result_lines = []
    
    for line in lines:
        if line not in unique_blocks:
            unique_blocks.add(line)
            result_lines.append(line)
    
    return '\n'.join(result_lines)

train.work_experience = train.work_experience.apply(remove_duplicate_blocks)

train['work_experience'] = train['work_experience'].astype(str).apply(lambda x: x[:-700])
train['position'] = train['position'].astype(str).apply(lambda x: x[:100])
train['key_skills'] = train['key_skills'].astype(str).apply(lambda x: x[:200])
train['salary'] = train['salary'].astype(str).apply(lambda x: x[:200])

train['text'] = (
    'age:\n' + train['age'].astype(str) +
    '\n\ncity:\n' + train['city'].astype(str) +
    '\n\nwork_experience:\n' + train['work_experience'] +
    '\n\nposition:\n' + train['position'] +
    '\n\nkey_skills:\n' + train['key_skills'] +
    '\n\nclient_name:\n' + train['client_name'].astype(str) +
    '\n\nsalary:\n' + train['salary']
)

train = train[['text', 'grade_proof']]
train = train.rename(columns={'grade_proof': 'label'})

train.label = train.label.apply(lambda x: 1 if x == 'подтверждён' else 0)

counts = train.label.value_counts().to_dict()
total_samples = sum(counts.values())
num_classes = len(counts)
class_weights = {cls: total_samples / (num_classes * count) for cls, count in counts.items()}

train, test = train_test_split(train, test_size=0.1, stratify=train['label'], random_state=52)

ds_train = Dataset.from_pandas(train)
ds_test = Dataset.from_pandas(test)

def encode(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=config.max_length)

ds_train = ds_train.map(encode, batched=True)
ds_test = ds_test.map(encode, batched=True)

class FocalLoss(nn.Module):
    '''
    Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    '''
    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 0.,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        '''
        Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        '''
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0., device=x.device, dtype=x.dtype)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class FocalTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.class_weights = class_weights.cuda() if torch.cuda.is_available() else class_weights
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        loss_foc = FocalLoss(alpha=self.class_weights, gamma=2)
        loss = loss_foc(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss

class TimeLimitCallback(TrainerCallback):
    def __init__(self, time_limit_hours):
        self.time_limit_seconds = time_limit_hours * 3600
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.time_limit_seconds:
            control.should_training_stop = True

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    y_true = eval_preds.label_ids
    y_pred = eval_preds.predictions.argmax(-1)
    y_proba = eval_preds.predictions[:, 1]

    return {
        'roc_auc': roc_auc_score(y_true=y_true, y_score=y_proba),
        'f1 (minor class)': f1_score(y_true=y_true, y_pred=y_pred),  
    }

def get_optimizer(model, backbone_LR):
    head_LR = backbone_LR * 10
    
    backbone_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters())
    
    optimizer = AdamW([
        {'params': backbone_params, 'lr': backbone_LR},
        {'params': head_params, 'lr': head_LR},
    ])
    
    return optimizer

optimizer = get_optimizer(model, backbone_LR=config.lr)

num_training_steps = (
    len(ds_train) // training_args.per_device_train_batch_size
) * training_args.num_train_epochs

num_warmup_steps = int(0.05 * num_training_steps)

scheduler = get_scheduler(
    'cosine', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

trainer = FocalTrainer(
    args=training_args, 
    model=model,
    #optimizers=(optimizer, scheduler),
    tokenizer=tokenizer,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    class_weights=list(class_weights.values()),
    callbacks=[TimeLimitCallback(time_limit_hours=11)]
)

trainer.train()
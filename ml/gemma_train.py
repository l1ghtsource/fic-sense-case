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
    BitsAndBytesConfig,
    Gemma2PreTrainedModel,
    Gemma2Model,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase, 
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    get_scheduler
)
from bitsandbytes.optim import AdamW8bit

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

VER = 'l4-2nd-try' # version of exp
USE_DEOTT_TOKENIZER = True  # https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527596

HAVE_GOOD_GPU = True

os.environ['WANDB_API_KEY'] = 'no please'
os.environ['WANDB_PROJECT'] = 'FIC Gemma2-9b'
os.environ['WANDB_NOTES'] = f'FIC Gemma2-9b LoRA Training VER-{VER}'
os.environ['WANDB_NAME'] = f'ft-gemma2-fic-ver-{VER}'

@dataclass
class Config:
    output_dir: str = 'output'
    checkpoint: str = 'unsloth/gemma-2-9b-it-bnb-4bit'
    num_labels: int = 2
    hdim: int = 3584
    max_length: int = 1024
    optim_type: str = 'adamw_8bit'
    per_device_train_batch_size: int = 2  # 4
    gradient_accumulation_steps: int = 4  # 2
    per_device_eval_batch_size: int = 4  # 8
    n_epochs: int = 1  # 2
    freeze_layers: int = 0 if HAVE_GOOD_GPU else 16
    lr: float = 2e-4  # 1e-5, 5e-5, 2e-5
    warmup_steps: int = 60
    lora_r: int = 64  # 128, 1024
    lora_alpha: float = 16 # lora_r * 2
    lora_dropout: float = 0.05
    lora_bias: str = 'none'
    
config = Config() # mb try to set learning rate of the backbone = alpha/rank * head_LRя

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
    save_steps=100,
    #optim=config.optim_type,
    fp16=True,
    #bf16=False,
    learning_rate=config.lr,
    #warmup_steps=config.warmup_steps,
    #lr_scheduler_type='cosine',  # 'cosine' or 'linear' or 'constant' (default is 'linear')
)

lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    task_type=TaskType.SEQ_CLS,
)

tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
tokenizer.add_eos_token = True
tokenizer.padding_side = 'right'

class CustomGemma2ForSequenceClassification(Gemma2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config)
        self.score = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(config.hidden_size, config.hidden_size // 2),
            torch.nn.Dropout(0.1),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_size // 2, config.num_labels),
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return super().forward(
            input_ids, 
            attention_mask, 
            position_ids, 
            past_key_values, 
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict
        )

model = Gemma2ForSequenceClassification.from_pretrained( #CustomGemma2ForSequenceClassification
    config.checkpoint,
    num_labels=config.num_labels,
    torch_dtype=torch.float16,
    device_map='auto',
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

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

counts = train.grade_proof.value_counts().to_dict()
total_samples = sum(counts.values())
num_classes = len(counts)
class_weights = {cls: total_samples / (num_classes * count) for cls, count in counts.items()}

train, test = train_test_split(train, test_size=0.1, stratify=train['grade_proof'], random_state=52)

ds_train = Dataset.from_pandas(train)
ds_test = Dataset.from_pandas(test)

ds = Dataset.from_pandas(train)

class CustomTokenizer:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, batch: dict) -> dict:
        age = [f'\n\n<age>: {t}' for t in batch['age']]
        city = [f'\n\n<city>: {t}' for t in batch['city']]
        work_experience = [f'\n\n<work_experience>: {t}' for t in batch['work_experience']][:-700]
        position = [f'<position>: {t}' for t in batch['position']][:100]
        key_skills = [f'\n\n<key_skills>: {t}' for t in batch['key_skills']][:200]
        client_name = [f'\n\n<client_name>: {t}' for t in batch['client_name']]
        salary = [f'\n\n<salary>: {t}' for t in batch['salary']][:200]
        
        texts = [
            a + c + we + p + ks + cn + s
            for a, c, we, p, ks, cn, s in zip(
                age,
                city,
                work_experience,
                position,
                key_skills,
                client_name,
                salary
            )
        ]
            
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        
        labels = []
        for grade_proof in batch['grade_proof']:
            if grade_proof == 'не подтверждён':
                label = 0
            elif grade_proof == 'подтверждён':
                label = 1
            labels.append(label)
            
        return {**tokenized, 'labels': labels}

# chris deott version, try to truncate from left side
class DeottCustomTokenizer:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_text(self, age, city, work_experience, position, key_skills, client_name, salary):
        rounds = [
            f'<start_of_turn>age\n{age}\n\ncity\n{city}\n\n'
            + f'work_experience\n{work_experience[:-700]}\n\n'
            + f'position\n{position[:100]}\n\n'
            + f'key_skills\n{key_skills[:200]}\n\n'
            + f'client_name\n{client_name}\n\n'
            + f'salary\n{salary[:200]}<end_of_turn>'
        ]
        
        tmp = '\n'.join(rounds)
        for k in range(len(rounds)):
            tmp = '\n'.join(rounds[k:])
            if len(self.tokenizer(tmp)['input_ids'] ) < self.max_length: 
                break
        
        return tmp
        
    def __call__(self, batch: dict) -> dict:
        texts = [
            self.prepare_text(a, c, we, p, ks, cn, s)
            for a, c, we, p, ks, cn, s in zip(
                batch['age'], 
                batch['city'], 
                batch['work_experience'], 
                batch['position'], 
                batch['key_skills'], 
                batch['client_name'], 
                batch['salary'], 
            )
        ]
        
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        
        labels = []
        for grade_proof in batch['grade_proof']:
            if grade_proof == 'не подтверждён':
                label = 0
            elif grade_proof == 'подтверждён':
                label = 1
            labels.append(label)
            
        return {**tokenized, 'labels': labels}

if USE_DEOTT_TOKENIZER:
    encode = DeottCustomTokenizer(tokenizer, max_length=config.max_length)
else:
    encode = CustomTokenizer(tokenizer, max_length=config.max_length)

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

def get_optimizer(model, head_LR, alpha, rank):
    backbone_LR = alpha / rank * head_LR
    
    backbone_params = list(model.base_model.model.model.parameters())
    head_params = list(model.score.parameters())
    
    optimizer = AdamW8bit([
        {'params': backbone_params, 'lr': backbone_LR},
        {'params': head_params, 'lr': head_LR},
    ])
    
    return optimizer

optimizer = get_optimizer(model, head_LR=config.lr, alpha=config.lora_alpha, rank=config.lora_r)

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
    optimizers=(optimizer, scheduler),
    tokenizer=tokenizer,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    class_weights=list(class_weights.values()),
    callbacks=[TimeLimitCallback(time_limit_hours=11)]
)

trainer.train()
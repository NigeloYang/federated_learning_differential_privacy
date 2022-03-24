import warnings

warnings.simplefilter("ignore")

import zipfile
import urllib.request
import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.utils.batch_memory_manager import BatchMemoryManager

import transformers
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers.data.processors.utils import InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features

# 加载数据集
STANFORD_SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
DATA_DIR = "../data/"


# 第一次运行下载的数据集容易出现提取报错，然后就直接手动解压,加入if条件就可以直接在第二次运行的时候不报错
# 提取出错原因：SNLI数据集的压缩文件"snli_1.0.zip"里面有两个路径为“snli_1.0\Icon\r”和“’__MACOSX/snli_1.0/._Icon\r’”的文件
def download_and_extract(dataset_url, data_dir):
  print("Downloading and extracting ...")
  if os.path.exists(os.path.join(data_dir, "snli_1.0")):
    print('解压完成')
  else:
    filename = "../data/snli.zip"
    urllib.request.urlretrieve(dataset_url, filename)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
      zip_ref.extractall(path=data_dir)
    os.remove(filename)
    print("Completed!")


download_and_extract(STANFORD_SNLI_URL, DATA_DIR)
snli_folder = os.path.join(DATA_DIR, "snli_1.0")
print(f'文件目录 \n  {os.listdir(snli_folder)}')

# 查看数据集
train_path = os.path.join(snli_folder, 'snli_1.0_train.txt')
test_path = os.path.join(snli_folder, 'snli_1.0_test.txt')

df_train = pd.read_csv(train_path, sep='\t')
df_test = pd.read_csv(test_path, sep='\t')

print(f'查看数据集属性名称 \n {df_train.columns}')
print('查看数据集前5行 \n', df_train[['sentence1', 'sentence2', 'captionID', 'pairID']].head())

# 建立模型
# BERT（来自 Transformers 的双向编码器表示）是各种 NLP 任务的最先进方法。 它使用 Transformer 架构，严重依赖预训练的概念
model_name = "bert-base-cased"
config = BertConfig.from_pretrained(model_name, num_labels=3, )
tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False, )
model = BertForSequenceClassification.from_pretrained("bert-base-cased", config=config, )

trainable_layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
total_params = 0
trainable_params = 0

for p in model.parameters():
  p.requires_grad = False
  total_params += p.numel()

for layer in trainable_layers:
  for p in layer.parameters():
    p.requires_grad = True
    trainable_params += p.numel()

print(f"Total parameters count: {total_params}")  # ~108M
print(f"Trainable parameters count: {trainable_params}")  # ~7M

# 预处理数据集
LABEL_LIST = ['contradiction', 'entailment', 'neutral']
MAX_SEQ_LENGHT = 128


def _create_examples(df, set_type):
  """ Convert raw dataframe to a list of InputExample. Filter malformed examples"""
  examples = []
  for index, row in df.iterrows():
    if row['gold_label'] not in LABEL_LIST:
      continue
    if not isinstance(row['sentence1'], str) or not isinstance(row['sentence2'], str):
      continue
    
    guid = f"{index}-{set_type}"
    examples.append(
      InputExample(guid=guid, text_a=row['sentence1'], text_b=row['sentence2'], label=row['gold_label']))
  return examples


def _df_to_features(df, set_type):
  """ Pre-process text. This method will:
  1) tokenize inputs
  2) cut or pad each sequence to MAX_SEQ_LENGHT
  3) convert tokens into ids

  The output will contain:
  `input_ids` - padded token ids sequence
  `attention mask` - mask indicating padded tokens
  `token_type_ids` - mask indicating the split between premise and hypothesis
  `label` - label
  """
  examples = _create_examples(df, set_type)
  
  # 向后兼容 old transformers versions
  legacy_kwards = {}
  from packaging import version
  if version.parse(transformers.__version__) < version.parse("2.9.0"):
    legacy_kwards = {
      "pad_on_left": False,
      "pad_token": tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
      "pad_token_segment_id": 0,
    }
  
  return glue_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    label_list=LABEL_LIST,
    max_length=MAX_SEQ_LENGHT,
    output_mode="classification",
    **legacy_kwards,
  )


def _features_to_dataset(features):
  """ Convert features from `_df_to_features` into a single dataset"""
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
  all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
  all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
  dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
  
  return dataset


train_features = _df_to_features(df_train, "train")
test_features = _df_to_features(df_test, "test")

train_dataset = _features_to_dataset(train_features)
test_dataset = _features_to_dataset(test_features)

# 选择尺寸
BATCH_SIZE = 32
MAX_PHYSICAL_BATCH_SIZE = 8

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

# 训练
# 选择合适的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')
model = model.to(device)

# Set the model to train mode (HuggingFace models load in eval mode)
model = model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8)

EPOCHS = 3
LOGGING_INTERVAL = 5000  # once every how many steps we run evaluation cycle and report metrics
EPSILON = 7.5
DELTA = 1 / len(train_dataloader)  # Parameter for privacy accounting. Probability of not achieving privacy guarantees


def accuracy(preds, labels):
  return (preds == labels).mean()


# define evaluation cycle
def evaluate(model):
  model.eval()
  
  loss_arr = []
  accuracy_arr = []
  
  for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    
    with torch.no_grad():
      inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[3]}
      
      outputs = model(**inputs)
      loss, logits = outputs[:2]
      
      preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
      labels = inputs['labels'].detach().cpu().numpy()
      
      loss_arr.append(loss.item())
      accuracy_arr.append(accuracy(preds, labels))
  
  model.train()
  return np.mean(loss_arr), np.mean(accuracy_arr)


# 加入噪音模块
MAX_GRAD_NORM = 0.1
privacy_engine = PrivacyEngine()

model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
  module=model,
  optimizer=optimizer,
  data_loader=train_dataloader,
  target_delta=DELTA,
  target_epsilon=EPSILON,
  epochs=EPOCHS,
  max_grad_norm=MAX_GRAD_NORM,
)

for epoch in range(1, EPOCHS + 1):
  losses = []
  
  with BatchMemoryManager(
      data_loader=train_dataloader, max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, optimizer=optimizer
  ) as memory_safe_data_loader:
    for step, batch in enumerate(tqdm(memory_safe_data_loader)):
      optimizer.zero_grad()
      
      batch = tuple(t.to(device) for t in batch)
      inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2], 'labels': batch[3]}
      
      outputs = model(**inputs)  # output = loss, logits, hidden_states, attentions
      
      loss = outputs[0]
      loss.backward()
      losses.append(loss.item())
      
      optimizer.step()
      
      if step > 0 and step % LOGGING_INTERVAL == 0:
        train_loss = np.mean(losses)
        eps = privacy_engine.get_epsilon(DELTA)
        
        eval_loss, eval_accuracy = evaluate(model)
        
        print(
          f"Epoch: {epoch} | Step: {step} | Train loss: {train_loss:.3f} | Eval loss: {eval_loss:.3f} | "
          f"Eval accuracy: {eval_accuracy:.3f} | ɛ: {eps:.2f}"
        )

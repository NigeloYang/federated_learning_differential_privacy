# -*- coding: utf-8 -*-
# @Time    : 2024/4/29


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

import transformers
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers.data.processors.utils import InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features


# # 建立模型
# BERT（来自 Transformers 的双向编码器表示）是各种 NLP 任务的最先进方法。 它使用 Transformer 架构，严重依赖预训练的概念
def create_model():
    config = BertConfig.from_pretrained('./bert_model/bert_base_cased', num_labels=3, )
    tokenizer = BertTokenizer.from_pretrained("./bert_model/bert_base_cased", do_lower_case=False, )
    model = BertForSequenceClassification.from_pretrained("./bert_model/bert_base_cased", config=config)
    
    return model, tokenizer


# 第一次运行下载的数据集容易出现提取报错，然后就直接手动解压,加入if条件就可以直接在第二次运行的时候不报错
# 提取出错原因：SNLI数据集的压缩文件"snli_1.0.zip"里面有两个路径为“snli_1.0\Icon\r”和“’__MACOSX/snli_1.0/._Icon\r’”的文件
def download_and_extract(dataset_url, data_dir):
    print("Downloading and extracting ...")
    if os.path.exists(os.path.join(data_dir, "snli_1.0")):
        print('已解压完成,开始预处理数据')
    else:
        print('解压中......')
        filename = "./data/snli.zip"
        urllib.request.urlretrieve(dataset_url, filename)
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(path=data_dir)
        os.remove(filename)
        print('解压完成')


# 加载数据集
def get_dataset(tokenizer, batch_size):
    STANFORD_SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    DATA_DIR = "./data/"
    
    download_and_extract(STANFORD_SNLI_URL, DATA_DIR)
    
    # 读取数据集
    df_train = pd.read_csv('./data/snli_1.0/snli_1.0_train.txt', sep='\t')
    df_test = pd.read_csv('./data/snli_1.0/snli_1.0_test.txt', sep='\t')
    
    print(f'查看数据集属性名称 \n {df_train.columns}')
    print('数据集标签 \n', df_train['gold_label'].value_counts())
    print('查看数据集前5行 \n', df_train.head(5))
    
    # 预处理数据集
    label_list = ['contradiction', 'entailment', 'neutral']
    MAX_SEQ_LENGHT = 128
    
    def _create_examples(df, set_type):
        """ Convert raw dataframe to a list of InputExample. Filter malformed examples"""
        examples = []
        for index, row in df.iterrows():
            if row['gold_label'] not in label_list:
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
            label_list=label_list,
            max_length=MAX_SEQ_LENGHT,
            output_mode="classification",
            **legacy_kwards,
        )
    
    # 转换为数据集
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
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)
    
    return train_dataloader, test_dataloader


# 训练
def train(model, train_loader, optimizer, privacy_engine, epoch, delta, device):
    model.train()
    losses = []
    acc = 0
    total = 0
    for idx, batch in enumerate(tqdm(train_loader)):
        batch = tuple(t.to(device) for t in batch)
        
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2],
                  'labels': batch[3]}
        
        total += len(inputs['labels'])
        
        outputs = model(**inputs)  # output = loss, logits, hidden_states, attentions
        
        # 计算精度
        logits = logits[1]
        
        # preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        # labels = inputs['labels'].detach().cpu().numpy()
        # acc += (preds == labels).mean()
        
        acc += (torch.sum(torch.argmax(logits, dim=1) == inputs['labels'])).item()
        
        # 计算损失
        loss = outputs[0]
        losses.append(loss.item())
        
        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (idx + 1) % 20 == 0:
            epsilon = privacy_engine.get_epsilon(delta)
            
            print(
                '| Train | Global Round: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f} | Epsilon: {:3.0f} | Delta: {:.6f}'.format(
                    epoch, 100. * (idx + 1) / len(train_loader), acc / total * 100, np.mean(losses), epsilon, delta))


# define evaluation cycle
def Test(model, test_loader):
    model.eval()
    
    losses = []
    acc = 0
    total = 0
    
    for batch in test_loader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2],
                      'labels': batch[3]}
            
            total += len(inputs['labels'])
            
            outputs = model(**inputs)
            
            # 计算精度
            logits = logits[1]
            
            # preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            # labels = inputs['labels'].detach().cpu().numpy()
            # acc += (preds == labels).mean()
            
            acc += (torch.sum(torch.argmax(logits, dim=1) == inputs['labels'])).item()
            
            # 计算损失
            loss = outputs[0]
            losses.append(loss.item())
            
    print('| Test |Acc: {:>3.0f}% | Loss: {:.3f} '.format(acc / total * 100, np.mean(losses)))


if __name__ == '__main__':
    # 选择合适的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device is {device}')
    
    # 定义训练参数
    batch_size = 32
    epochs = 3
    
    # 获取模型 查看模型参数数量
    model, tokenizer = create_model()
    model = model.to(device)
    
    trainable_layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
    total_params = 0
    trainable_params = 0
    
    for p in model.parameters():
        p.requires_grad = False
        total_params += p.numel()
    print(f"Total parameters count: {total_params}")  # ~108M
    
    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()
    print(f"Trainable parameters count: {trainable_params}")  # ~7M
    
    # 获取数据
    train_loader, test_loader = get_dataset(tokenizer, batch_size)
    
    # 定义噪声模块参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8)
    epsilon = 7.5
    delta = 1 / len(train_loader)  # Parameter for privacy accounting. Probability of not achieving privacy guarantees
    sigma = 1.0
    max_grad_norm = 0.1
    
    # 加入 opacus 进行 DP 训练
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False
    )
    
    for epoch in range(epochs):
        train(model, train_loader, optimizer, privacy_engine, epoch+1, delta, device)
    
    Test(model, test_loader)

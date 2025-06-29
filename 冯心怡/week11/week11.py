!pip install evaluate
!pip install seqeval
from transformers import AutoModelForTokenClassification, AutoTokenizer,DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import evaluate  
from datasets import load_dataset
import numpy as np
# 加载hf中dataset
ds = load_dataset('doushabao4766/msra_ner_k_V3')
ds
for items in ds['train']:
    print(items['tokens'])
    print(items['ner_tags'])
    break
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
### 实体映射字典

'O':0   
'B-PER':1   
'I-PER':2   
'B-LOC':3   
'I-LOC':4   
'B-ORG':5   
'I-ORG':6   
# 验证tag标签数量
tags_id = set()
for items in ds['train']:
    tags_id.update(items['ner_tags'])
    
tags_id
# entity_index
entites = ['O'] + list({'PER', 'LOC', 'ORG'})
tags = ['O']
for entity in entites[1:]:
    tags.append('B-' + entity.upper())
    tags.append('I-' + entity.upper())

entity_index = {entity:i for i, entity in enumerate(entites)}

entity_index
tags
def data_input_proc(item):
    # 文本已经分为字符，且tag索引也已经提供
    # 所以数据预处理反而简单
    # 导入已拆分为字符的列表，需要设置参数is_split_into_words=True
    input_data = tokenizer(item['tokens'], 
                           truncation=True,
                           add_special_tokens=False, 
                           max_length=512, 
                           is_split_into_words=True,
                           return_offsets_mapping=True)
    
    labels = [lbl[:512] for lbl in item['ner_tags']]
    input_data['labels'] = labels
    return input_data

ds1 = ds.map(data_input_proc, batched=True)
ds1.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
for item in ds1['train']:
    print(item)
    break
### 构建模型对象
id2lbl = {i:tag for i, tag in enumerate(tags)}
lbl2id = {tag:i for i, tag in enumerate(tags)}

model = AutoModelForTokenClassification.from_pretrained('bert-base-chinese', 
                                                        num_labels=len(tags),
                                                        id2label=id2lbl,
                                                        label2id=lbl2id)
model
### 模型训练 TrainningArguments
args = TrainingArguments(
    output_dir="msra_ner_train",  # 模型训练工作目录（tensorboard，临时模型存盘文件，日志）
    num_train_epochs = 3,    # 训练 epoch
    # save_safetensors=False,  # 设置False保存文件可以通过torch.load加载
    per_device_train_batch_size=32,  # 训练批次
    per_device_eval_batch_size=32,
    report_to='tensorboard',  # 训练输出记录
    eval_strategy="epoch",
)
### 模型训练 Trainer
# metric 方法
def compute_metric(result):
    # result 是一个tuple (predicts, labels)
    
    # 获取评估对象
    seqeval = evaluate.load('seqeval')
    predicts,labels = result
    predicts = np.argmax(predicts, axis=2)
    
    # 准备评估数据
    predicts = [[tags[p] for p,l in zip(ps,ls) if l != -100]
                 for ps,ls in zip(predicts,labels)]
    labels = [[tags[l] for p,l in zip(ps,ls) if l != -100]
                 for ps,ls in zip(predicts,labels)]
    results = seqeval.compute(predictions=predicts, references=labels)

    return results
# import evaluate 
# evaluate.load('seqeval')
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
trainer = Trainer(
    model,
    args,
    train_dataset=ds1['train'],
    eval_dataset=ds1['test'],
    data_collator=data_collator,
    compute_metrics=compute_metric
)
**模型训练**
trainer.train()
**模型推理**
from transformers import pipeline
pipeline = pipeline('token-classification', 'msra_ner_train/checkpoint-500')
pipeline('双方确定了今后发展中美关系的指导方针')
pipeline('双方确定了今后发展中日关系的指导方针')
pipeline('双方确定了今后发展美日关系的指导方针')

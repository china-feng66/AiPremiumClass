import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import pandas as pd
from tqdm import tqdm 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter()

df = pd.read_excel('/kaggle/input/jd_comment_with_label/jd_comment_data.xlsx')

df.head()
df.columns
filterd= df['评价内容(content)'] != "此用户未填写评价内容"

data_df = df[filterd][['评价内容(content)','评分（总分5分）(score)']]

data_df.head()
print(type(data_df))
data = data_df.values
data
train ,test = train_test_split(data)
print(train.shape)
print(test.shape)
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-bert-wwm', mirror="tuna")
# DataLoader创建方法
def warp_data(batch_data):
    comments, lables = [],[]
    for bdate in batch_data:
        comments.append(bdate[0])
        lables.append(int(bdate[1])-1)  # 标签取值[0-4]
    
    # 转换模型输入数据
    input_data = tokenizer(comments, return_tensors='pt', padding=True, truncation=True, max_length=512)
    labels_data = torch.tensor(lables)
    
    return input_data, labels_data

train_dl = DataLoader(train, batch_size=20, shuffle=True, collate_fn = warp_data)
test_dl = DataLoader(test, batch_size=20, shuffle=False, collate_fn = warp_data)
def warp_data(batch_data):
    comments,labels = [],[]
    for bdate in batch_data:
        comments.append(bdate[0])
        labels.append(int(bdate[1])-1)
    input_data = tokenizer(comments,return_tensors='pt',padding=True,truncation=True,max_length=512)
    labels_data = torch.tensor(labels)
    return input_data,labels_data
    
train_dl = DataLoader(train,batch_size = 20,shuffle=True,collate_fn=warp_data)
test_dl = DataLoader(test,batch_size = 20 , shuffle=True,collate_fn=warp_data)
for item in test_dl:
    print(item)
    break

# model_1 模型微调 Supervised Fine Tuning
# model_2 迁移学习 Transfer Learning 冻结bert
# model_1 = AutoModelForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm', num_labels=5)
# model_2 = AutoModelForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm', num_labels=5)
model_1 = AutoModelForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm', num_labels=5)
model_2 = AutoModelForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm', num_labels=5)
model_1 = model_1.to(device)
model_2 = model_2.to(device)
model_2.bert.trainable = False
# loss、optim
loss_fn1 = nn.CrossEntropyLoss()
optim1 = Adam(model_1.parameters(), lr=1e-4)

loss_fn2 = nn.CrossEntropyLoss()
optim2 = Adam(model_2.parameters(), lr=1e-3)

# 不冻结
model1_train_loss_cnt = 0

for epoch in range(5):
    pbar = tqdm(train_dl)
    for input_data, labels_data in pbar:
        datas = { k:v.to(device) for k,v in input_data.items() }
        labels = labels_data.to(device)
        
        result = model_1(**datas)
        loss = loss_fn1(result.logits, labels)
        
        pbar.set_description(f'epoch:{epoch} train_loss:{loss.item():.4f}')

        writer.add_scalar("Fine Tuning Train Loss", loss, model1_train_loss_cnt)
        model1_train_loss_cnt += 1
        
        loss.backward()
        optim1.step()
        
        model_1.zero_grad()


torch.save(model_1.state_dict(),'model_1.pt')
        
# 冻结
model2_train_loss_cnt = 0

for epoch in range(5):
    pbar = tqdm(train_dl)
    for input_data, labels_data in pbar:
        datas = { k:v.to(device) for k,v in input_data.items() }
        labels = labels_data.to(device)
        
        result = model_2(**datas)
        loss = loss_fn2(result.logits, labels)
        
        pbar.set_description(f'epoch:{epoch} train_loss:{loss.item():.4f}')

        writer.add_scalar("Transfer Learning Train Loss", loss, model2_train_loss_cnt)
        model2_train_loss_cnt += 1
        
        loss.backward()
        optim2.step()
        
        model_2.zero_grad()


torch.save(model_2.state_dict(),'model_2.pt')
model_1.eval()
model_2.eval()
pbar = tqdm(test_dl)


correct1, correct2 = 0,0

for input_data, labels_data in pbar:
    datas = { k:v.to(device) for k,v in input_data.items() }
    labels = labels_data.to(device)

    with torch.no_grad():
        result1 = model_1(**datas)
        result2 = model_2(**datas)

    predict1 = torch.argmax(result1.logits, dim=-1)
    predict2 = torch.argmax(result2.logits, dim=-1)

    correct1 += (predict1 == labels).sum()
    correct2 += (predict1 == labels).sum()

# model_1.load_state_dict()

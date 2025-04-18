# %%
# 资料\DMSC.csv
import csv
ds_comments = []
sum_ = 0
with open('./DMSC.csv','r') as file:
        reader = csv.DictReader(file)
        for row in reader:
                sum_ += 1
print(sum_)

# %%
# 加载训练语料
import csv
import jieba
ds_comments = []
with open('./DMSC.csv','r') as file:
        reader = csv.DictReader(file)
        
        for read in reader:
            vote = int(read['Star'])
            if vote in [1,2]:
                words = jieba.lcut(read['Comment'])
                ds_comments.append((words,1))
            if vote in [4,5]:
                words = jieba.lcut(read['Comment'])
                ds_comments.append((words,0))
print(len(ds_comments))

# %%
ds_comments = [c for c in ds_comments if len(c[0]) in range(90, 100)]
import pickle

with open('comments.pkl', 'wb') as f:
    pickle.dump(ds_comments, f)
len(ds_comments)

# %%
with open('comments.pkl','rb') as f:
    ds_comments = pickle.load(f)
        

# %%
from torch.nn.utils.rnn import pad_sequence
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# %%
# 构建词典表
def build_from_doc(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])
    vocab = ['PAD','UNK'] + list(vocab)
    w2idx = {word:indx for indx,word in enumerate(vocab)}
    return w2idx
class Comments_Classifier(nn.Module):
    def __init__(self, vocab_size,embedding_dim,hidden_size,num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)
    def forward(self,input_ids):
        embedded = self.embedding(input_ids)
        # output: (batch_size, seq_len, hidden_size)
        output, (hidden, _) = self.rnn(embedded)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出
        return output

# %%
# comments_data = ds_comments
# 构建词汇表
vocab = build_from_doc(comments_data)
print(len(vocab))
emb = nn.Embedding(len(vocab),100)
# 自定义转换方法即回调函数
# 填充长度并把评论和评分转化为张量
def convert_data(batch_data):
    comments,votes=[],[]
    for comment,vote in batch_data:
        comments.append(torch.tensor([vocab.get(word,vocab['UNK']) for word in comment]))
        votes.append(vote)
    commt = pad_sequence(comments,batch_first=True,padding_value=vocab['PAD'])# 填充相同长度
    labels = torch.tensor(votes)
    return commt,labels
# 构建DataLoader
dataloader = DataLoader(comments_data,batch_size=2048,shuffle=True,collate_fn=convert_data)

vocab_size = len(vocab)
embedding_dim = 100
hidden_size = 128
num_classes = 2

model = Comments_Classifier(len(vocab), embedding_dim, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 训练模型
num_epochs = 1
for epoch in range(num_epochs):
    for i, (cmt, lbl) in enumerate(dataloader):
        cmt = cmt
        lbl = lbl

        # 前向传播
        outputs = model(cmt)
        loss = criterion(outputs, lbl)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')


# %%
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 

def build_from_doc(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])

    vocab =  ['PAD','UNK'] + list(vocab)  
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx

class Comments_Classifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # padding_idx=0
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        
        # input_ids: (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        # output: (batch_size, seq_len, hidden_size)
        output, (hidden, _) = self.rnn(embedded)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出
        return output

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练语料
    with open('comments.pkl','rb') as f:
        comments_data = pickle.load(f)

    # 构建词汇表
    vocab = build_from_doc(comments_data)
    print('词汇表大小:', len(vocab))

    # 所有向量集合 Embedding（词嵌入）
    emb = nn.Embedding(len(vocab), 100) # 词汇表大小，向量维度

    # 自定义数据转换方法(callback function)回调函数
    # 该函数会在每个batch数据加载时被调用
    def convert_data(batch_data):
        comments, votes = [],[]
        # 分别提取评论和标签
        for comment, vote in batch_data:
            comments.append(torch.tensor([vocab.get(word, vocab['UNK']) for word in comment]))
            votes.append(vote)
        
        # 将评论和标签转换为tensor
        commt = pad_sequence(comments, batch_first=True, padding_value=0)  # 填充为相同长度
        labels = torch.tensor(votes)
        # 返回评论和标签
        return commt, labels

    # 通过Dataset构建DataLoader
    dataloader = DataLoader(comments_data, batch_size=512, shuffle=True, 
                            collate_fn=convert_data)

    # 构建模型
    # vocab_size: 词汇表大小
    # embedding_dim: 词嵌入维度
    # hidden_size: LSTM隐藏层大小
    # num_classes: 分类数量
    vocab_size = len(vocab)
    embedding_dim = 200
    hidden_size = 128
    num_classes = 5

    model = Comments_Classifier(len(vocab), embedding_dim, hidden_size, num_classes)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (cmt, lbl) in enumerate(dataloader):
            cmt = cmt.to(device)
            lbl = lbl.to(device)

            # 前向传播
            outputs = model(cmt)
            loss = criterion(outputs, lbl)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')




from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformer_model import Seq2SeqTransformer
import torch
# 1. 构建词典
def build_vocab(token_lists):
    vocab = {'<pad>': 0, '<s>': 1, '</s>': 2}
    idx = 3
    for tokens in token_lists:
        for t in tokens:
            if t not in vocab:
                vocab[t] = idx
                idx += 1
    return vocab

# 2. 数据集
class MyDataset(Dataset):
    def __init__(self, enc_tokens, dec_tokens, enc_vocab, dec_vocab):
        self.enc_tokens = enc_tokens
        self.dec_tokens = dec_tokens
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab

    def __len__(self):
        return len(self.enc_tokens)

    def __getitem__(self, idx):
        enc = [self.enc_vocab[t] for t in self.enc_tokens[idx]]
        dec = [self.dec_vocab[t] for t in self.dec_tokens[idx]]
        return torch.tensor(enc, dtype=torch.long), torch.tensor(dec, dtype=torch.long)

# 3. collate_fn
def collate_fn(batch):
    enc_batch, dec_batch = zip(*batch)
    enc_batch = pad_sequence(enc_batch, batch_first=True, padding_value=0)
    
    # dec_in: 去掉最后一个
    dec_in = [dec[:-1] for dec in dec_batch]
    # dec_out: 去掉第一个
    dec_out = [dec[1:] for dec in dec_batch]
    
    dec_in = pad_sequence(dec_in, batch_first=True, padding_value=0)
    # dec_out: 去掉第一个
    dec_out = pad_sequence(dec_out, batch_first=True, padding_value=0)
    return enc_batch, dec_in, dec_out

# 4. mask
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)
    return mask

if __name__ == '__main__':
    corpus= "人生得意须尽欢，莫使金樽空对月"
    chs = list(corpus)
    
    enc_tokens, dec_tokens = [],[]

    for i in range(1,len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)

    # 构建词典
    enc_vocab = build_vocab(enc_tokens)
    dec_vocab = build_vocab(dec_tokens)
    inv_dec_vocab = {v: k for k, v in dec_vocab.items()}

    # 构建数据集和dataloader
    dataset = MyDataset(enc_tokens, dec_tokens, enc_vocab, dec_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 模型参数
    d_model = 32
    nhead = 4
    num_enc_layers = 2
    num_dec_layers = 2
    dim_forward = 64
    dropout = 0.1
    enc_voc_size = len(enc_vocab)
    dec_voc_size = len(dec_vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqTransformer(d_model, nhead, num_enc_layers, num_dec_layers, dim_forward, dropout, enc_voc_size, dec_voc_size).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    # 训练
    for epoch in range(50):
        model.train()
        total_loss = 0
        for enc_batch, dec_in, dec_out in dataloader:
            enc_batch, dec_in, dec_out = enc_batch.to(device), dec_in.to(device), dec_out.to(device)
            tgt_mask = generate_square_subsequent_mask(dec_in.size(1)).to(device)
            enc_pad_mask = (enc_batch == 0)
            dec_pad_mask = (dec_in == 0)
            logits = model(enc_batch, dec_in, tgt_mask, enc_pad_mask, dec_pad_mask)
            # logits: [B, T, V]  dec_out: [B, T]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'transformer.pth')
    print("模型已保存。")

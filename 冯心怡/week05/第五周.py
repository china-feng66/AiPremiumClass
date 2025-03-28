# %%
from tqdm import tqdm
fixed = open('doubanbook_top250_comments_fixed.txt','w')

lines = [line for line in open('E:\Mi\第五周\doubanbook_top250_comments.txt','r')]
for i,line in enumerate(tqdm(lines)):
    if i==0:
        fixed.write(line)
        pre_line =''
        continue

    terms = line.split('\t')
    if terms[0] == pre_line.split('\t')[0]:
        if len(pre_line.split('\t'))==6:
            fixed.write(pre_line + '\n')
            pre_line = line.strip()
        else:
            pre_line = ""
    else:
        if len(terms)==6:
            pre_line = line.strip()
        else:
            pre_line +=line.strip()

fixed.close()





# %%
import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bm25 import bm25
def load_data(filename):
    book_comments={}
    with open(filename,'r') as f:
        read = csv.DictReader(f,delimiter='\t')
        for line in read:
            book = line['book']
            comment = line['body']
            comment_words=jieba.cut(comment)
            if book =='':continue
            # 收集评论集合
            book_comments[book]=book_comments.get(book,[])
            book_comments[book].extend(comment_words)
    return book_comments
if __name__ == "__main__":
    stop_words=[line.strip() for line in open("E:\Mi\第五周\stopwords.txt",'r',encoding='utf-8')]
    book_comments=load_data("doubanbook_top250_comments_fixed.txt")
    # 提取书名和评论文本
    book_names=[]
    book_comms=[]
    for book,comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)
    # 计算tf-idf
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([' '.join(comm) for comm in book_comms])

    # 输入要推荐的图书名称
    book_list = list(book_comments.keys())
    book_name = input("请输入要推荐的图书名称：")
    book_idx = book_names.index(book_name)

    # 计算相似度
    similarity=cosine_similarity(tfidf_matrix)
    # 输入要推荐的图书名称
    book_list = list(book_comments.keys())
    book_name = input("请输入要推荐的图书名称：")
    book_idx = book_names.index(book_name)
    # 获取与输入图书最相似的图书
    recommend_book_index = np.argsort(-similarity[book_idx])[1:11]
    # 输出推荐的图书
    for idx in recommend_book_index:
        print(f"《{book_names[idx]}》\t 相似度{similarity[book_idx][idx]:.2f}")

# %%
import csv
import jieba
import numpy as np
from bm25 import bm25

def load_data(filename):
    book_comments = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for line in reader:
            book = line['book']
            comment = line['body']
            comment_words = [word for word in jieba.cut(comment) if word.strip()]
            if book == '' or not comment_words:
                continue
            book_comments.setdefault(book, []).extend(comment_words)
    return book_comments

if __name__ == "__main__":
    # 加载停用词
    stop_words = set(line.strip() for line in open("E:\Mi\第五周\stopwords.txt", 'r', encoding='utf-8'))
    
    # 加载数据
    book_comments = load_data("doubanbook_top250_comments_fixed.txt")
    
    # 准备数据
    book_names = list(book_comments.keys())
    book_comms = [list(filter(lambda x: x not in stop_words, words)) 
                 for words in book_comments.values()]
    
    # 计算BM25相似度矩阵
    similarity_matrix = bm25(book_comms)
    
    # 获取推荐
    book_name = input("请输入要推荐的图书名称：")
    try:
        book_idx = book_names.index(book_name)
        
        # 获取相似度行并排序
        similarities = similarity_matrix[book_idx]
        normalized_matrix = np.linalg.norm(similarity_matrix, axis=1)
        similarities = similarities / normalized_matrix
        sorted_indices = np.argsort(similarities)[::-1]  # 降序排序
        
        # 获取推荐索引（跳过自己，取前10）
        recommend_indices = sorted_indices[sorted_indices != book_idx][:10]
        
        # 输出结果
        print(f"\n基于BM25算法，与《{book_name}》最相似的图书：")
        for idx in recommend_indices:
            print(f"《{book_names[idx]}》\t相似度：{similarities[idx]}")
            
    except ValueError:
        print("未找到该图书，请检查输入！")

# %%
lines[0].split('\t')


# %%
len(lines[1].split('\t'))


# %%
lines[9]

# %%
tqdm(lines)

# %%
lines = [line for line in open('E:\Mi\第五周\doubanbook_top250_comments.txt','r')]
lines[9]
lines[10]

# %%
# from fasttext import FastText
# wv_file = ''
# wv_vectors = FastText.load_model(wv_file)
input_file = "E:/Mi/第五周/data_train.txt"
output_file = "E:/Mi/第五周/data_train_cleaned.txt"

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    
    for line in fin:
        # 处理多标签情况
        parts = line.strip().split()
        labels = [p for p in parts if p.startswith('__label__')]
        text = ' '.join([p for p in parts if not p.startswith('__label__')])
        
        # 规范格式：多个标签用空格分隔，最后接文本
        fout.write(f"{' '.join(labels)} {text}\n")








# %%
import fasttext
import os

def validate_file(path):
    """验证文件是否可用于FastText训练"""
    if not os.path.exists(path):
        raise ValueError(f"文件不存在: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line.startswith('__label__'):
                raise ValueError("首行必须以__label__开头")
    except UnicodeDecodeError:
        raise ValueError("文件编码不是UTF-8，请转换为UTF-8格式")
    except Exception as e:
        raise ValueError(f"文件读取失败: {str(e)}")

def train_model():
    path = r"E:/Mi/data_train_cleaned.txt"
    
    try:
        validate_file(path)
        model = fasttext.train_supervised(
            input=path,
            lr=0.1,
            epoch=25,
            wordNgrams=2,
            verbose=2
        )
        print("训练成功！")
        return model
    except Exception as e:
        print(f"训练失败: {str(e)}")
        return None

if __name__ == "__main__":
    model = train_model()
    if model:
        model.save_model("model.bin")

# %%
from fasttext import FastText
model = FastText.load_model("model.bin")
model.predict("Which baking dish is best to bake a banana bread ?")
model.predict("Which baking dish is best to bake a banana bread ?", k=3)
model.predict([
    "Which baking dish is best to bake a banana bread ?",
    "Why not put knives in the dishwasher?"
],k=3)

# %%
# "E:\Mi\书籍\大模型导论 数据集\第11章\short_news.txt"
import fasttext
import fasttext.util
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 示例文档数据（替换为您的自定义文本）
documents = [
    "盗墓笔记讲述了吴邪的冒险故事",
    "鬼吹灯描写了胡八一寻找精绝古城的经历",
    "三体是刘慈欣创作的科幻小说",
    "白夜行是东野圭吾的推理小说代表作",
    "机器学习需要大量文本数据进行训练"
]

# 1. 准备训练数据（写入临时文件）
with open("corpus.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc + "\n")

# 2. 训练FastText模型（同时学习子词信息）
ft_model = fasttext.train_unsupervised(
    input="corpus.txt",
    model='skipgram',  # 使用skipgram算法
    dim=100,           # 词向量维度
    ws=5,              # 上下文窗口大小
    minCount=1,        # 最小词频
    epoch=20,          # 训练轮次
    wordNgrams=2       # 考虑子词组合
)

# 3. 保存和加载模型
ft_model.save_model("ft_model.bin")
ft = fasttext.load_model("ft_model.bin")

# 4. 获取词向量
word_vectors = {}
words = ["盗墓", "笔记", "科幻", "小说", "机器学习"]
for word in words:
    word_vectors[word] = ft.get_word_vector(word)

# 5. 计算词汇相关度（余弦相似度）
def calculate_similarity(word1, word2):
    vec1 = ft.get_word_vector(word1).reshape(1, -1)
    vec2 = ft.get_word_vector(word2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# 示例计算
test_pairs = [
    ("盗墓", "笔记"),
    ("科幻", "小说"),
    ("盗墓", "机器学习")
]

print("词汇相似度计算结果:")
for w1, w2 in test_pairs:
    sim = calculate_similarity(w1, w2)
    print(f"'{w1}'与'{w2}'的相似度: {sim:.4f}")

# 6. 查找最相似词（可选）
print("\n与'科幻'最相似的词:")
neighbors = ft.get_nearest_neighbors("科幻", k=3)
for score, word in neighbors:
    print(f"{word}: {score:.4f}")



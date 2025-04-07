import os
import math
import random
import jieba
import numpy as np

from collections import Counter


def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return set(stopwords)


def load_and_preprocess_comments(file_path):
    comments = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        prev_book_name = ''
        for i, line in enumerate(lines):
            if i == 0:
                continue

            # if i == 5000:
            #     break

            terms = line.split("\t")
            if len(terms) >= 6:
                if terms[0] == '':
                    terms[0] = prev_book_name
                if terms[0] not in comments:
                    comments[terms[0]] = []
                
                comment_text = "\t".join(terms[5:])
                comments[terms[0]].append(comment_text)
                prev_book_name = terms[0]
            else:
                comments[prev_book_name][-1] += terms[0]
            
    return comments


def chinese_tokenizer(text, stopwords):
    words = jieba.lcut(text)
    return [word for word in words if word not in stopwords]


def calculate_bm25(comments, stopwords, k1=1.5, b=0.75):
    N = len(comments)
    doc_lengths = {}
    avgdl = 0
    all_terms = []
    doc_term_freq = {}
    
    for book_name, comments_list in comments.items():
        all_terms_for_book = []
        for comment in comments_list:
            words = chinese_tokenizer(comment, stopwords)
            all_terms_for_book.extend(words)
            all_terms.extend(words)
        
        doc_lengths[book_name] = len(all_terms_for_book)
        avgdl += len(all_terms_for_book)
        doc_term_freq[book_name] = Counter(all_terms_for_book)
    
    avgdl /= N
    
    # IDF
    idf = {}
    
    unique_terms = set(all_terms)
    
    for term in unique_terms:
        doc_count_with_term = sum(1 for doc_terms in doc_term_freq.values() if term in doc_terms)
        idf[term] = math.log((N - doc_count_with_term + 0.5) / (doc_count_with_term + 0.5))
    
    # BM25
    bm25_scores = {}
    
    for book_name, term_freq in doc_term_freq.items():
        bm25_scores[book_name] = {}
        doc_length = doc_lengths[book_name]
        
        for term, tf in term_freq.items():
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))
            bm25_scores[book_name][term] = idf.get(term, 0) * (numerator / denominator)
    
    return bm25_scores, list(unique_terms)


def get_book_bm25_vector(BM25, book_name, all_terms):
    vector = []
    for term in all_terms:
        vector.append(BM25.get(book_name, {}).get(term, 0))
    return np.array(vector)


def my_cosine_similarity(vec1, vec2):
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    vec1_normalized = vec1 / norm_vec1
    vec2_normalized = vec2 / norm_vec2
    
    return np.dot(vec1_normalized, vec2_normalized)


def get_most_similar_books(bm25, all_terms, selected_book, top_n=5):
    book_names = list(bm25.keys())
    selected_vector = get_book_bm25_vector(bm25, selected_book, all_terms)
    
    similarities = []
    for book in book_names:
        if book != selected_book:
            book_vector = get_book_bm25_vector(bm25, book, all_terms)
            similarity = my_cosine_similarity(selected_vector, book_vector)
            similarities.append((book, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def main():
    dataset_path = './邪王真眼/dataset/'
    comments_file = os.path.join(dataset_path, 'doubanbook_top250_comments.txt')
    stopwords_file = os.path.join(dataset_path, 'stopwords.txt')

    comments = load_and_preprocess_comments(comments_file)
    stopwords = load_stopwords(stopwords_file)
    
    selected_book = random.choice(list(comments.keys()))
    selected_book = '盗墓笔记'
    print(f"Selected Book: {selected_book}")

    bm25, all_terms = calculate_bm25(comments, stopwords)

    similar_books = get_most_similar_books(bm25, all_terms, selected_book, top_n=5)

    print("Most similar books:")
    for book, similarity in similar_books:
        print(f"Book: {book}, Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()

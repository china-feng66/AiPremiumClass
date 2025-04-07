import os
import math
import random
import jieba
import numpy as np

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def calculate_tfidf(comments, stopwords):
    doc_term_freq = {}
    all_terms = []

    for book_name, comments_list in comments.items():
        all_terms_for_book = []
        for comment in comments_list:
            words = chinese_tokenizer(comment, stopwords)
            all_terms_for_book.extend(words)
            all_terms.extend(words)
        
        doc_term_freq[book_name] = Counter(all_terms_for_book)

    idf = {}
    total_docs = len(comments)
    unique_terms = set(all_terms)
    
    for term in unique_terms:
        doc_count_with_term = sum(1 for doc_terms in doc_term_freq.values() if term in doc_terms)
        idf[term] = math.log((total_docs + 1) / (1 + doc_count_with_term))

    tfidf = {}

    for book_name, term_freq in doc_term_freq.items():
        tfidf[book_name] = {}
        for term, tf in term_freq.items():
            tfidf[book_name][term] = tf * idf.get(term, 0)

    return tfidf, all_terms


def get_book_tfidf_vector(tfidf, book_name, all_terms):
    vector = []
    for term in all_terms:
        vector.append(tfidf.get(book_name, {}).get(term, 0))
    return np.array(vector)


def my_cosine_similarity(vec1, vec2):
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    vec1_normalized = vec1 / norm_vec1
    vec2_normalized = vec2 / norm_vec2
    
    return np.dot(vec1_normalized, vec2_normalized)


def get_most_similar_books(tfidf, all_terms, selected_book, top_n=5):
    book_names = list(tfidf.keys())
    selected_vector = get_book_tfidf_vector(tfidf, selected_book, all_terms)
    
    similarities = []
    for book in book_names:
        if book != selected_book:
            book_vector = get_book_tfidf_vector(tfidf, book, all_terms)
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

    manual = True

    if manual == True:
        tfidf, all_terms = calculate_tfidf(comments, stopwords)

        similar_books = get_most_similar_books(tfidf, all_terms, selected_book, top_n=5)

        print("Most similar books:")
        for book, similarity in similar_books:
            print(f"Book: {book}, Similarity: {similarity:.4f}")
    
    else:
        vec = TfidfVectorizer(stop_words=list(stopwords))
        tokenized_comments = [' '.join(chinese_tokenizer(' '.join(comms), stopwords)) for comms in comments.values()]
        tfidf = vec.fit_transform(tokenized_comments)
        similar = cosine_similarity(tfidf)

        selected_book_index = list(comments.keys()).index(selected_book)
        similarity_scores = similar[selected_book_index]
        similarity_scores[selected_book_index] = -1
        most_similar_books_indices = similarity_scores.argsort()[-5:][::-1]
        
        print("Most similar books:")
        for index in most_similar_books_indices:
            book_name = list(comments.keys())[index]
            similarity = similarity_scores[index]
            print(f"Book: {book_name}, Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()

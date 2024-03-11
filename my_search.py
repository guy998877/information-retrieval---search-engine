import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
nltk.download('stopwords')
import requests
from inverted_index_gcp import *
import math
from collections import Counter, OrderedDict
import re


index_src_title = "index"
index_src_body = "index_body"

base_dir_title = "postings_gcp/"
base_dir_body = "postings_gcp_body/"
bucket_name = "yuval_206542839"

index_title = InvertedIndex.read_index(base_dir_title, f'{index_src_title}', bucket_name)
index_body = InvertedIndex.read_index(base_dir_body, f'{index_src_body}', bucket_name)
index_id_title = InvertedIndex.read_index(base_dir_body, f'{"index_id_title"}', bucket_name)
dict_id_title = index_id_title.dict_id_title


def preprocess_query(query):
    """
    Preprocesses the query by removing stopwords, stemming, and tokenization.
    """

    # Stopwords list
    stop_words = set(stopwords.words('english'))

    # Stemmer
    stemmer = PorterStemmer()

    # lower case
    query = query.lower()

    # Remove non-alphabetic characters
    query = re.sub(r'[^a-zA-Z\s]', '', query)

    # Tokenize
    tokens = query.split()

    # Remove stopwords and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return tokens


def query_to_vector(query_tokens, index):
    """
    Transforms a query into a vector using TF-IDF weights from the inverted index.
    :param query_tokens: List of tokens in the query
    :param index: The inverted index object
    :return: A dictionary representing the query vector, with terms as keys and TF-IDF as values.
    """
    # Calculate TF for the query
    query_tf = Counter(query_tokens)
    query_vector = {}

    for term, tf in query_tf.items():
        # Calculate TF-IDF, assuming the document contains the term
        if term in index.df:
            query_vector[term] = tf

    return query_vector


def cosine_similarity(query_vector, index):
    """
    Calculates the cosine similarity between a query vector and document vectors,
    where document vectors are represented by posting lists of tokens that appear both in the documents and the query.
    :param query_vector: The query vector represented as a dictionary.
    :param posting_lists: A dictionary of posting lists, with terms as keys and lists of (doc_id, tf_idf) as values.
    :return: A list of tuples, each tuple being (doc_id, cosine_similarity_score).
    """
    # Initialize document scores and norms
    doc_scores = Counter()
    doc_norms = Counter()

    # Calculate scores for each term in the query
    for term, query_weight in query_vector.items():
        if term in index.df:
            posting = index.read_a_posting_list("", term, "yuval_206542839")
            for doc_id, tf_idf in posting:
                doc_scores[doc_id] += query_weight * tf_idf
                doc_norms[doc_id] += tf_idf ** 2

    # Calculate query norm
    sum_total = 0
    for weight in query_vector.values():
        sum_total += weight ** 2
    query_norm = math.sqrt(sum_total)

    # Calculate final cosine similarity scores
    similarities = []
    for doc_id, score in doc_scores.items():
        doc_norm = math.sqrt(doc_norms[doc_id])
        if query_norm == 0 or doc_norm == 0:
            similarity = 0
        else:
            similarity = score / (query_norm * doc_norm)
        similarities.append((doc_id, similarity))

    return similarities[:1000]


def bm25_score(query_vector, index, k1=1.5, b=0.75):
    """
    Calculates the BM25 score between a query vector and document vectors.
    :param query_vector: The query vector represented as a dictionary with terms and their TF-IDF weights.
    :param index: The inverted index object containing document frequencies and other relevant data.
    :param k1: The scaling factor for term frequency. Typically between 1.2 and 2.0.
    :param b: The document length normalization factor. Typically close to 0.75.
    :return: A list of tuples, each tuple being (doc_id, BM25_score).
    """
    doc_scores = Counter()

    for term, query_tf in query_vector.items():
        if term in index.df:
            posting = index.read_a_posting_list("", term, "yuval_206542839")
            for doc_id, doc_tf in posting:
                # IDF calculation for the term
                idf = math.log((index.N - index.df[term] + 0.5) / (index.df[term] + 0.5) + 1)

                # Term frequency normalization and scaling
                B = 1 - b + b * (index.dict_len[doc_id] / index.AVG)
                tf_scaled = (doc_tf * (k1 + 1)) / (doc_tf + k1 * B)
                # Accumulate the BM25 score for the document
                doc_scores[doc_id] += idf * tf_scaled

    # Convert scores to a sorted list of tuples (doc_id, score)
    sorted_scores = doc_scores.most_common()  # This will sort documents by their BM25 score in descending order

    return sorted_scores[:1000]


def return_result(page_ids, dict_id_title):
    """
    recieve list of page_ids
    return list (doc_id,title)
    """
    titles = []
    for page_id in page_ids:
        titles.append((str(page_id), dict_id_title[page_id]))

    return titles


def calculate_result(bm25_scores, cosine_similarity_scores, weight_TITLE, weight_BODY):
    # Convert lists to dictionaries

    bm25_dict = {key: value for key, value in bm25_scores}
    cosine_similarity_dict = {key: value for key, value in cosine_similarity_scores}

    # Normalize BM25 scores using min-max normalization
    min_bm25 = bm25_scores[-1][1]
    max_bm25 = bm25_scores[0][1]
    normalized_bm25_dict = {doc_id: (score - min_bm25) / (max_bm25 - min_bm25) for doc_id, score in bm25_dict.items()}

    # Initialize a dictionary to store the final weighted scores
    final_scores = {}

    # Calculate the weighted score for documents appearing in either or both dictionaries
    all_doc_ids = set(bm25_dict.keys()).union(set(cosine_similarity_dict.keys()))
    for doc_id in all_doc_ids:
        bm25_score = normalized_bm25_dict.get(doc_id, 0)
        cosine_score = cosine_similarity_dict.get(doc_id, 0)
        final_scores[doc_id] = weight_BODY * bm25_score + weight_TITLE * cosine_score

    # Sort the final scores dictionary by score in descending order to see the highest ranked documents first
    sorted_final_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # Display the sorted final scores
    return sorted_final_scores

def my_search(query):
    """
    Performs the search operation, including query preprocessing, document retrieval,
    and document ranking.
    """

    tokens = preprocess_query(query)
    query_vector_title = query_to_vector(tokens, index_title)
    query_vector_body = query_to_vector(tokens, index_body)

    sim_cosine = cosine_similarity(query_vector_title, index_title)
    sim_bm25 = bm25_score(query_vector_body, index_body, k1=1.5, b=0.75)

    result = calculate_result(sim_bm25, sim_cosine, 0.3, 0.7)

    res_lst = [doc_id for doc_id, _ in result]

    res_short = res_lst[:100]

    res = return_result(res_short, dict_id_title)
    return res

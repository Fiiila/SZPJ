from vector_search import import_documents, prepare_documents, apply_stemming, load_topics, compute_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from pathlib import Path
import re


if __name__ == '__main__':

    # Hyperparameters
    useStemming = True  # swith for comparing with and without stemming
    document_path = Path('./SZPJ_SP1_collection/documents')  # path to dictionary with source content files
    queries_path = Path('./SZPJ_SP1_collection/query_devel.xml')  # path to file containing queries
    output_path = Path('./vector_search_output.txt')  # savepath for output of search
    qrel_path = Path('./SZPJ_SP1_collection/cacm_devel.rel')  # file for computing score


    # Ensure that nltk has required files downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # import collection of documents
    names, documents = import_documents(document_path)
    print(f"Loaded {len(documents)} documents.")

    # preprocess documents and group relevant content together
    prep_doc = prepare_documents(names, documents)
    print(f"\nRAW document content example for file {names[0]}\n"
          f"{documents[0]}")
    print(f"\nGrouped document content example for file {names[0]}\n"
          f"{prep_doc[names[0]]['content']}")

    # create log file with prepared content of all documents
    with open('log.txt', 'w', encoding='utf-8') as log:
        for i, n in enumerate(names):
            log.write(f'{i+1}\t {(prep_doc[n]["year"])}\t {(prep_doc[n]["month"])}\n')
            log.write(f'Content: {prep_doc[n]["content"]}\n')

    # create data for tfidf vectorizer with removing necessary interpunction
    data = [re.sub(r"[.,!?#$%&'*+]", '', ' '.join(prep_doc[n]["content"])) for n in names]
    # use stemming
    if useStemming:
        data = apply_stemming(data)

    # Specify vectorizer object and transform data
    tfidf = TfidfVectorizer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words='english')
    sparse_doc_term_matrix = tfidf.fit_transform(data)

    # Load queries
    queries = load_topics(queries_path)
    if useStemming:
        stemmed_queries = apply_stemming([i[1] for i in queries])
        queries = [[queries[i][0], stemmed_queries[i]] for i in range(len(queries))]

    # create output
    with open(output_path, 'w', encoding='utf-8') as output:
        for q in queries:
            q_temp = tfidf.transform([q[1]])
            sim_temp = cosine_similarity(sparse_doc_term_matrix, q_temp).flatten()
            sim_temp_best_args = sim_temp.argsort(axis=None)[::-1][:100]
            for s in sim_temp_best_args:
                output.write(f"{q[0]}\t{names[s].split('.')[0]}\t{sim_temp[s]}\n")
    print("\nOutput created.")

    # compute MAP score
    print("\nComputing MAP...\n")
    compute_score(qrel=qrel_path, scored=output_path)


import os
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk import PorterStemmer
import nltk


def import_documents(path):
    """Method for importing all documents into memory while remowing unnecessary information

    :param path: Path to the folder with all required documents to be searched in
    :type path: Path
    :return: two lists contain names of individual documents and according document raw content
    :rtype: list, list
    """
    documents = []  # empty list for documents content
    names = os.listdir(path)  # get list of filenames in provided folders

    # cycle through filenames
    for doc in names:
        res_doc = []  # resulting document after skipping lines without important content
        # read content of whole document
        with open(path.joinpath(doc), 'r', encoding='utf-8') as d:
            temp_doc = d.readlines()

        # filter out lines in document
        for line in temp_doc:
            # skip lines containing someting in arrow brackets <>
            if re.search(r'<\S+>', line):
                continue
            # skip lines containing just numbers and whitespace characters
            elif re.fullmatch(r'(\d+\s)+', line):
                continue
            # skip line containing probably time and date of access
            elif re.search(r'\w+,\s(\d{4})\s+(\d+:\d+)\s+(AM|PM)', line):
                continue
            # append lines not skipped by conditions above
            else:
                res_doc.append(line)
        # append filtered document lines into list of documents
        documents.append(res_doc)
    return names, documents


def prepare_documents(names, documents):
    """Method for preprocessing documents and putting it into more structured form into groups (title, abstract...)

    :param names: list of document filenames
    :type names: list
    :param documents: list of document lines
    :type documents: list
    :return: dictionary with individual documents, keys are filenames and prepared document content is under content
    :rtype: dict
    """
    # dictionary for month replacement by numbers for planned filtering - not used
    months = {'january': 1,
              'february': 2,
              'march': 3,
              'april': 4,
              'may': 5,
              'june': 6,
              'july': 7,
              'august': 8,
              'september': 9,
              'october': 10,
              'november': 11,
              'december': 12
              }
    # dictionary for processed documents
    processed_documents = {}
    # compiled regex
    findLineEnd = re.compile(r'.+\n$')
    findDate = re.compile(
        r'(january|february|march|april|may|june|july|august|september|october|november|december)\W+(\d{4})',
        flags=re.IGNORECASE)

    # cycle through document contents
    for idx, doc in enumerate(documents):
        grouping = []  # list for all groups in one document
        temp_group = []  # list for just one group in document
        prev_line = ''  # previous line for algorithm for dividing into groups

        # cycle through lines in one document
        for i, line in enumerate(doc):
            # detect empty line after some content
            if line == '\n' and findLineEnd.search(prev_line):
                grouping.append(' '.join([l.strip() for l in temp_group]))
                temp_group.clear()
                prev_line = line
                continue
            # detect content on current line and empty previous line
            if line != '\n' and prev_line == '\n':
                temp_group.clear()
            # append line into temporal group
            temp_group.append(line)
            prev_line = line

        # add document into dictionary
        processed_documents[names[idx]] = {}
        processed_documents[names[idx]]['content'] = grouping

        # optional step for additional dividing into other categories
        for i,g in enumerate(grouping):
            if i == 0:
                processed_documents[names[idx]]['title'] =g
            elif re.search(findDate, g):
                spl = re.search(pattern=findDate, string=g)
                processed_documents[names[idx]]['month'] = spl.group(1)
                processed_documents[names[idx]]['year'] = spl.group(2)

    return processed_documents


def load_topics(path):
    """Load topics to be searched from file

    :param path: path to file containing searches
    :type path: Path
    :return: list of number and string to be searched
    :rtype: list
    """
    with open(Path(path), 'r', encoding='utf-8') as file:
        lines = file.readlines()
    topics = []
    temp_topic = []
    for line in lines:
        # identify DOC tag explaining start of new question
        if re.search(r'(<DOC>)', line):
            temp_topic.clear()
        # identify number of question
        elif re.search(r'(<DOCNO>)\s+(\d+)\s+(<\/DOCNO>)', line):
            temp_topic.append(re.search(r'(<DOCNO>)\s+(\d+)\s+(<\/DOCNO>)', line).group(2))
        # identify question of ending DOC tag explaining end of question and append current topic into list of topics
        elif re.search(r'(</DOC>)', line):
            topics.append([temp_topic[0], ' '.join(temp_topic[1:])])
        # identify empty line
        elif re.fullmatch(r'\n', line):
            continue
        # append line the topic
        else:
            temp_topic.append(line.strip())
    return topics

def apply_stemming(data):
    """Method for applying stemming to limit number of words in final vectorized searching

    :param data: list of content to individual documents
    :type data: list
    :return: list of input data processed through stemming
    :rtype: list
    """
    stemmer = PorterStemmer() #SnowballStemmer("english")
    res = []
    temp_sentence = []
    # cycle through sentences in all data provided
    for sentence in data:
        temp_sentence.clear()
        for word in nltk.word_tokenize(sentence):
            temp_sentence.append(stemmer.stem(word))
        res.append(' '.join(temp_sentence))
    return res


import math
def compute_score(qrel, scored):
    # qrel = 'cacm_devel.rel'  #### file with relevance judgments
    # scored = '../vector_search_output.txt'  # 'vzor_vystupu_vyhledavaciho_programu.txt'         #### file with retrieved documents

    # qrel = 'toy.rel'
    # scored = 'toy.out'

    rel_docs = {}
    retrieved_docs = {}

    AP = {}

    with open(qrel, 'r') as relevant_files:  ##  read the list of relevant files
        for line in relevant_files:
            line.strip()
            items = line.split(" ")

            if items[0] not in rel_docs.keys():
                rel_docs[items[0]] = []

            rel_docs[items[0]].append(items[2])
            # print line

    with open(scored, 'r') as retrieved_files:
        for line in retrieved_files:
            line.strip()
            items = line.split("\t")

            if items[0] not in retrieved_docs.keys():
                retrieved_docs[items[0]] = []

            retrieved_docs[items[0]].append(items[1])

            # print line

    ### compute average precisions for individual topics first
    acc_MAP = 0

    for topic in retrieved_docs:
        acc_AP = 0
        number_of_relevant = 0

        for position in range(100):
            if retrieved_docs[topic][position] in rel_docs[topic]:
                number_of_relevant += 1
                this_point_P = number_of_relevant / (position + 1)
                acc_AP += this_point_P
                # print retrieved_docs[topic][position] + " " + str(this_point_P)

        if (number_of_relevant == 0):
            AP[topic] = 0
        else:
            AP[topic] = acc_AP / (len(rel_docs[topic]))

        acc_MAP += AP[topic]
        print_output = "AP for topic " + topic + " is " + str(AP[topic])
        print(print_output)

    MAP = acc_MAP / len(retrieved_docs)

    print_map = "**** MAP is " + str(MAP)
    print(print_map)


if __name__ == '__main__':
    # for testing purposes

    document_path = Path('./SZPJ_SP1_collection/documents')
    names, documents = import_documents(document_path)
    print(f"Loaded {len(documents)} documents")
    print(documents[0])
    prep_doc = prepare_documents(names, documents)
    print(names[0])
    print(prep_doc[names[0]]['content'])
    # print(prep_doc[names[0]]['date'])
    with open('log.txt', 'w') as log:
        for i,n in enumerate(names):
            log.write(f'{i}\t {(prep_doc[n]["year"])}\t {(prep_doc[n]["month"])}\n')
            log.write(f'{prep_doc[n]["content"]}\n')

    data = [re.sub(r"[.,!?#$%&'*+]", '', ' '.join(prep_doc[n]["content"])) for n in names] #!#$%&'*+-.^_`|~:
    data = apply_stemming(data)
    # Specify vectorizer object
    tfidf = TfidfVectorizer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words='english')
    sparse_doc_term_matrix = tfidf.fit_transform(data)
    # print(sparse_doc_term_matrix.shape)
    #
    # query = ['What articles exist which deal with TSS (Time Sharing System), an operating system for IBM computers?']
    # q = tfidf.transform(query)
    # dense_q = q.toarray()
    # print(dense_q)
    # sim = cosine_similarity(sparse_doc_term_matrix, q)
    # print(type(sim))
    # sim_best = sim.flatten().argsort(axis=None)[::-1][:100]
    with open('./vector_search_output.txt', 'w') as output:
        # output.write(', '.join(tfidf.get_feature_names_out()))
        queries = load_topics(Path('./SZPJ_SP1_collection/query_devel.xml'))
        stemmed_queries = apply_stemming([i[1] for i in queries])
        queries = [[queries[i][0] ,stemmed_queries[i]] for i in range(len(queries))]
        for q in queries:
            q_temp = tfidf.transform([q[1]])
            sim_temp = cosine_similarity(sparse_doc_term_matrix, q_temp).flatten()
            sim_temp_best_args = sim_temp.argsort(axis=None)[::-1][:100]
            for s in sim_temp_best_args:
                output.write(f"{q[0]}\t{names[s].split('.')[0]}\t{sim_temp[s]}\n")
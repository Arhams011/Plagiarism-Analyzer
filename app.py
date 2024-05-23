import os
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Load student files
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

def vectorize_with_doc2vec(Text):
    # Tokenize the text into words
    tokenized_text = [doc.split() for doc in Text]

    # Tag documents for training Doc2Vec
    tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(tokenized_text)]

    # Train Doc2Vec model
    doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, epochs=20, dm=1)
    doc2vec_model.build_vocab(tagged_data)
    doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

    # Get document vectors
    doc_vectors = [doc2vec_model.docvecs[str(i)] for i in range(len(tokenized_text))]

    return doc_vectors

def similarity(doc1, doc2):
    # Compute the cosine similarity between two document vectors
    return cosine_similarity([doc1, doc2])

# Vectorize the student notes using Doc2Vec
vectors = vectorize_with_doc2vec(student_notes)
plagiarism_results = set()

def check_plagiarism():
    global vectors
    for i, vector_a in enumerate(vectors):
        for j, vector_b in enumerate(vectors):
            if i != j:
                sim_score = similarity(vector_a, vector_b)[0][1]
                student_pair = sorted((student_files[i], student_files[j]))
                score = (student_pair[0], student_pair[1], sim_score)
                plagiarism_results.add(score)
    return plagiarism_results

# Check for plagiarism and print results
for data in check_plagiarism():
    print(data)

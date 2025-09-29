import json
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer


def remove_digits(s):
    return re.sub(r'\d+', '', s)


CASE_NAME = ["cpu1", "memory1", "network3", "query1", "workload1"]

all_text = []
text_group = []
GROUP_LENGTH = 50
REDUNDANT_NUM = 10
for case in CASE_NAME:
    for test_index in range(1, 21):
        if os.path.exists(f"IoTDB_result/{case}_test{test_index}_group={GROUP_LENGTH}_operator.log"):
            document_text = ""
            fr = open(f"IoTDB_result/{case}_test{test_index}_group={GROUP_LENGTH}_operator.log", "r")
            lines = fr.readlines()
            for line in lines:
                document_text += line
            document_text = remove_digits(document_text)
            all_text.append(document_text)
            for i in range(REDUNDANT_NUM):
                curr_window_lines = lines[i:len(lines) - i]
                text = ""
                for line in curr_window_lines:
                    text += line
                text = remove_digits(text)
                text_group.append(text)

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(all_text)
tfidf_matrix = vectorizer.transform(text_group)
tfidf_matrix_array = tfidf_matrix.toarray()

# Compute embeddings for each file
curr_index = 0
for case in CASE_NAME:
    for test_index in range(1, 21):
        for i in range(REDUNDANT_NUM):
            embeddings = tfidf_matrix_array[curr_index].tolist()
            with open(f"IoTDB_result/{case}_test{test_index}_embedding_group={GROUP_LENGTH}_tf-idf_group{i}.json",
                      "w") as f:
                json.dump(embeddings, f)
            curr_index += 1

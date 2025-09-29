import json
import math
import os

import fasttext

CASE_NAME = ["cpu1", "memory1", "network3", "query1", "workload1"]

SEGMENT_LENGTH = 100

fw = open(f"IoTDB_result/all_text.txt", "w")
GROUP_LENGTH = 50
for case in CASE_NAME:
    all_text = ""
    for test_index in range(1, 21):
        if os.path.exists(f"IoTDB_result/{case}_test{test_index}_group={GROUP_LENGTH}_operator.log"):
            fr = open(f"IoTDB_result/{case}_test{test_index}_group={GROUP_LENGTH}_operator.log", "r")
            lines = fr.readlines()
            for line in lines:
                all_text += line
    fw.write(all_text)
fw.flush()
fw.close()

DIM_LENGTH = 100
model = fasttext.train_unsupervised('IoTDB_result/all_text.txt', model='skipgram', dim=DIM_LENGTH)  # model入参可以更换为`cbow`
# Compute embeddings for each file
for case in CASE_NAME:
    for test_index in range(1, 21):
        file_path = f"IoTDB_result/{case}_test{test_index}_group={GROUP_LENGTH}_operator.log"
        print(file_path)
        if os.path.exists(file_path):
            with open(file_path, "r") as fr:
                lines = fr.readlines()
                segment_num = math.ceil(len(lines) / SEGMENT_LENGTH)
                for loop in range(segment_num):
                    segment_lines = lines[loop * SEGMENT_LENGTH: min(len(lines), (loop + 1) * SEGMENT_LENGTH)]
                    embeddings = []
                    for line in lines:
                        while "\n" in line:
                            line = line.replace("\n", "")
                        if len(line) > 0:
                            embedding = model.get_sentence_vector(line).tolist()
                            embeddings.append(embedding)
                    with open(f"IoTDB_result/{case}_test{test_index}_embedding_group={GROUP_LENGTH}_fasttext.json",
                              "w") as f:
                        json.dump(embeddings, f)

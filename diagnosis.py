import os.path

from scipy import stats
import json

CASE_NAME = ["cpu1", "memory1", "network3", "query1", "workload1"]

GROUP_LENGTH = 50
REDUNDANT_NUM = 10
CASE_NUM = 20

semantic_embedding = {}
for case in CASE_NAME:
    semantic_embedding[case] = []
    for test_index in range(1, CASE_NUM + 1):
        for i in range(REDUNDANT_NUM):
            if os.path.exists(
                    f"IoTDB_result/{case}_test{test_index}_embedding_group={GROUP_LENGTH}_fasttext_group{i}.json"):
                fr = open(f"IoTDB_result/{case}_test{test_index}_embedding_group={GROUP_LENGTH}_fasttext_group{i}.json",
                          "r")
                semantic_embedding[case].append(json.load(fr))

quantity_embedding = {}
for case in CASE_NAME:
    quantity_embedding[case] = []
    for test_index in range(1, CASE_NUM + 1):
        for i in range(REDUNDANT_NUM):
            if os.path.exists(
                    f"IoTDB_result/{case}_test{test_index}_embedding_group={GROUP_LENGTH}_tf-idf_group{i}.json"):
                fr = open(f"IoTDB_result/{case}_test{test_index}_embedding_group={GROUP_LENGTH}_tf-idf_group{i}.json",
                          "r")
                quantity_embedding[case].append(json.load(fr))

ratio = 1
LABEL_NUM = 5
TOP_NUM = 3
result = {}
for case in CASE_NAME:
    result[case] = {"TP": 0, "FN": 0, "FP": 0}
for case in CASE_NAME:
    for test_index in range(LABEL_NUM, CASE_NUM * REDUNDANT_NUM, 1):
        sim_list = []
        for label_case in CASE_NAME:
            for i in range(LABEL_NUM):
                semantic_sim = stats.pearsonr(semantic_embedding[label_case][i], semantic_embedding[case][test_index])[
                    0]
                quantity_sim = stats.pearsonr(quantity_embedding[label_case][i], quantity_embedding[case][test_index])[
                    0]
                sim_list.append((label_case, semantic_sim * ratio + quantity_sim * (1 - ratio)))
        sim_list.sort(key=lambda x: x[1], reverse=True)
        sim_list = sim_list[:TOP_NUM]
        label_num = {}
        for (label_case, _) in sim_list:
            if label_case not in label_num:
                label_num[label_case] = 0
            label_num[label_case] += 1
        res = max(label_num, key=lambda x: label_num[x])
        print(case + "->" + res)
        if case == res:
            result[case]["TP"] += 1
        else:
            result[case]["FN"] += 1
            result[res]["FP"] += 1

total_precision = 0
total_recall = 0
total_f1 = 0
for case in CASE_NAME:
    precision = result[case]["TP"] / (result[case]["TP"] + result[case]["FP"])
    recall = result[case]["TP"] / (result[case]["TP"] + result[case]["FN"])
    f1 = 2 * precision * recall / (precision + recall)
    print(f"{case}:{precision}-{recall}-{f1}")
    total_precision += precision
    total_recall += recall
    total_f1 += f1
print(f"macro: {total_precision / len(CASE_NAME)}-{total_recall / len(CASE_NAME)}-{total_f1 / len(CASE_NAME)}")

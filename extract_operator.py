import os
import re

GROUP_LENGTH = 50
CASE_NAME = ["cpu1", "memory1", "network3", "query1", "workload1"]

# Compute embeddings for each file
for case in CASE_NAME:
    for test_index in range(1, 21):
        file_path = f"IoTDB_result/{case}_test{test_index}_summary_group={GROUP_LENGTH}_gpt3.5.log"
        if os.path.exists(file_path):
            with open(file_path, "r") as fr:
                text = fr.read()
                while "\n" in text:
                    text = text.replace("\n", "")
                matches = re.findall(r'\[(.*?)]', text)
                with open(f"IoTDB_result/{case}_test{test_index}_group={GROUP_LENGTH}_operator.log",
                          "w") as f:
                    result = []
                    for match in matches:
                        if len(match) > 5:
                            result.append(match + "\n")
                    f.writelines(result)

import math
import time

import openai
from http import HTTPStatus
import dashscope

openai.api_key = "sk-key"
dashscope.api_key = "sk-key"


def gpt3_5(prompt, input_str):
    retry_time = 5
    while retry_time > 0:
        try:
            completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_str}
            ])
            content = completion.choices[0].message.content
            if content is None:
                return ""
            else:
                return content
        except Exception as e:
            print(e)
            retry_time -= 1
    return ""


def qwen(prompt, input_str):
    messages = [{'role': 'system', 'content': prompt},
                {'role': 'user', 'content': input_str}]
    response = dashscope.Generation.call(
        "qwen-turbo",
        messages=messages,
        # set the result to be "message" format.
        result_format='message',
    )
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


GROUP_LENGTH = 50
CASE_NAME = ["cpu1", "memory1", "network3", "query1", "workload1"]
for case in CASE_NAME:
    for test_index in range(1, 21):
        fr = open(f"IoTDB/{case}_test{test_index}.log", "r")
        lines = fr.readlines()
        fr.close()
        results = []
        for i in range(math.ceil(len(lines) / GROUP_LENGTH)):
            input_str = ""
            for j in range(GROUP_LENGTH):
                if i * GROUP_LENGTH + j < len(lines):
                    input_str += lines[i * GROUP_LENGTH + j] + "\n"
                else:
                    break
            result = gpt3_5(
                "Give a summary of the following system logs in 500 words, it will later be used to classify one kind of anomaly."
                " The summary needs and only needs to contain all operations performed by the system, wrapped with [] and separated by;"
                "For example: [initializing the authorizer];[registering to a cluster];[flush working memtables]",
                input_str)
            while "\n" in result:
                result = result.replace("\n", "")
            results.append(result)
            fw = open(f"IoTDB_result/{case}_test{test_index}_summary_group={GROUP_LENGTH}_gpt3.5.log", "a")
            fw.write(result + "\n")
            fw.flush()
            fw.close()
            print(f"IoTDB_result/{case}-{test_index}:Progress={i}/{math.ceil(len(lines) / GROUP_LENGTH)}")
            print(result)
            time.sleep(5)

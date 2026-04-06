import time

import torch
from transformers import pipeline

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# remove new lines (llm does not need it)
def compact_prompt(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


question = "What did Branislav do in ecx?"

final_prompt = """
<|system|>
You are an intent parser for questions about a person's CV. Your job is to convert the question into a JSON object with three fields:
- domain
- queryType
- parameters

Allowed domains:
EMPLOYMENT
SKILLS
PROFILE
PROJECTS

Allowed queryTypes:
FACT
LIST
SUMMARY
EXPLANATION

Allowed parameters:
year (integer)
yearPeriod (array of two integers)
company (string)
skills (array of strings)
current (boolean)

Rules:
- Only use the allowed domains and queryTypes.
- Extract parameters only if they appear in the question.
- If a parameter does not appear, do not include it.
- If no parameters exist, return an empty parameters object {}.
- Output MUST be valid JSON.
- Output ONLY the JSON object.
- Do not include explanations or text before or after the JSON.

Examples:

Question: Where does he work currently?
{"domain":"EMPLOYMENT","queryType":"FACT","parameters":{"current":true}}

Question: What did he do in IBM?
{"domain":"EMPLOYMENT","queryType":"FACT","parameters":{"company":"IBM"}}

Question: What skills does he have?
{"domain":"SKILLS","queryType":"LIST","parameters":{}}

Question: What did he do in 2013?
{"domain":"EMPLOYMENT","queryType":"FACT","parameters":{"year":2013}}

Question: What jobs did he have between 2015 and 2016?
{"domain":"EMPLOYMENT","queryType":"LIST","parameters":{"yearPeriod":[2015,2016]}}

Question: Summarize his profile.
{"domain":"PROFILE","queryType":"SUMMARY","parameters":{}}
<|end|>
<|user|>
Question: $QUESTION$
<|end|>
<|assistant|>
""".replace("$QUESTION$", question)

# llm_service = LLMService()
# answer = llm_service.answer(question, context)
start = time.perf_counter()
generator_pipeline = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", device=0)
response = generator_pipeline(compact_prompt(final_prompt))
#print(response)
print(f"query duration {time.perf_counter() - start}")
print(response[0]["generated_text"].split("<|assistant|>")[1].strip())

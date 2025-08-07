from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="ZyrexAn/DataScience-PromptEngineer-merged",
    device_map="auto"   # will use your RTX 3070
)

prompt = "How to load a CSV and plot age distribution in Python?"
result = pipe(prompt, max_new_tokens=200)
print(result[0]['generated_text'])

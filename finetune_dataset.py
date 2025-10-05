import json
from openai import OpenAI
from config import OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL

AMAZON_TRN_PATH = "dataset/LF-Amazon-1.3M/trn.json"
# JSONL -> Efficient for large datasets (line-by-line processing)
FINETUNE_DATASET_PATH = "dataset/finetune_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are a model specialized in answering questions about products.\n"    
    "Rules:\n"
    "1. The user will ask questions related to a product.\n"    
    "2. Your answer must be based only on the description given by the user.\n"
    "3. Never invent information beyond what is in the description.\n"
    "4. The answer must be clear, concise, and professional.\n"    
    "5. The questions will be in the format ['question 1', 'question 2', 'question 3']\n"
    "6. The answers will be in the a JSON valid format : { responses: ['answer 1', 'answer 2', 'answer 3']}\n"
)

QUESTION_TEMPLATES = [
    "What does the product '{title}' offer?",
    "What are the main features of '{title}'?",
    "Can you describe the product '{title}'?",
    "What are the advantages of '{title}'?",
    "Tell me more about '{title}'.",
    "What makes '{title}' special?",
    "Describe the functionalities of '{title}'.",
    "What can I expect when buying '{title}'?",
]

def load_amazon_trn(path):
    """
    Load the Amazon Titles dataset from a JSON file.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file if line.strip()]
    return data

def user_prompt_text(questions, content):    
    return f"""Answer the questions based on the description below.\n
Questions:\n
{questions}
\n
Description:\n
{content}
"""

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)
model = OPENAI_MODEL

def chat_response(prompt):
    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            { 
                "role": "user",
                "content": prompt
            },
        ],
    )
    content_response = response.choices[0].message.content
    return json.loads(content_response)["responses"]

def create_finetune_dataset(data, output_path, max_examples=-1):
    """
    Create a finetune dataset in JSONL format.
    """
    examples = []
    for item in data:
        title = item.get("title", "")
        content = item.get("content", "")

        if not title or not content or content.strip() == "":
            continue  # ignora produtos inválidos        
        
        questions = [template.format(title=title) for template in QUESTION_TEMPLATES]
       
        user_prompt = user_prompt_text(questions, content)
        try:                
            responses = chat_response(user_prompt)
        except Exception as e:                
            responses = [f"Error processing item with title '{title}': {e}"] * len(questions)

        example = {            
            "title": title,
            "content": content,
            "questions": questions,
            "responses": responses,
        }
        
        examples.append(example)
        
        if max_examples > 0 and len(examples) >= max_examples:
            break
    
    # Write to JSONL file
    with open(output_path, "w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example) + "\n")

MAX_TOTAL_EXAMPLES = 10000  # limit total examples for testing
# 10 -> 1m26,699 deepseek-chat (87s)
# 10 -> 0m43,718s gpt-3.5-turbo (44s)
# 10 -> 0m35,828s llama3.2:3b (36s)
data = load_amazon_trn(AMAZON_TRN_PATH)
create_finetune_dataset(data=data, output_path=FINETUNE_DATASET_PATH, max_examples=MAX_TOTAL_EXAMPLES)
# source env/bin/activate
# time python finetune_dataset.py

# 460.000
# 460.000/10 * 36 = 1.656.000 s
# 27.600 m
# 460 h
# 19 d
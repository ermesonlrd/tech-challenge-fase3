import json
import re
from openai import OpenAI
from config import OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
from dataset_utils import load_amazon_trn, clean_content, save_chunk, sanitize_text
from typing import Dict, List
import time

# Limites
CHUNK_SIZE = 1000         # Máx. de produtos por arquivo
MAX_TOTAL_EXAMPLES = -1   # limit total examples (-1 for all)

SYSTEM_PROMPT = (
    "You are a model specialized in answering questions about products.\n"    
    "Rules:\n"
    "1. The user will ask questions related to a product.\n"    
    "2. Your answer must be based only on the description given by the user.\n"
    "3. Never invent information beyond what is in the description.\n"
    "4. The answer must be clear, concise, and professional.\n"    
    "5. The questions will be in the format ['question 1', 'question 2']\n"
    "6. The answers will be in the a JSON valid format : { responses: ['answer 1', 'answer 2']}\n"
)

# QUESTION_TEMPLATES = [
#     "What does the product '{title}' offer?",
#     "What are the main features of '{title}'?",
#     "Can you describe the product '{title}'?",
#     "What are the advantages of '{title}'?",
#     "Tell me more about '{title}'.",
#     "What makes '{title}' special?",
#     "Describe the functionalities of '{title}'.",
#     "What can I expect when buying '{title}'?",
# ]

QUESTION_TEMPLATES = [
    "Can you describe the product '{title}'?",
    "What are the main features and advantages of '{title}'?"
]

def user_prompt_text(questions, content):    
    return f"""Answer the questions based on the description below.\n
Questions:\n
{questions}
\n
Description:\n
{content}
"""

def validate_and_fix_json(json_str: str) -> dict:
    """Tenta validar e corrigir JSON malformado."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        
        # Tenta algumas correções comuns
        fixed_json = json_str
        
        # Corrige aspas não escapadas
        fixed_json = re.sub(r'(?<!\\)"(?![,}\]\s])', '\\"', fixed_json)
        
        # Remove vírgulas extras antes de } ou ]
        fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
        
        # Tenta novamente
        try:
            return json.loads(fixed_json)
        except json.JSONDecodeError:
            # Se ainda não funcionar, retorna estrutura vazia
            print("Could not fix JSON, returning empty structure")
            return {"responses": []}

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)
model = OPENAI_MODEL


def chat_response(prompt: str, max_retries: int = 2) -> List[str]:
    """Chama a API com tratamento robusto de erros e retry."""
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
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
                max_tokens=1024,  # Aumentado para evitar truncamento
                temperature=0.7,  # Adiciona um pouco de variabilidade
            )
            
            content_response = response.choices[0].message.content
            
            if not content_response:
                raise ValueError("Empty response from API")
            
            # Limpa a resposta antes de processar
            cleaned_content = sanitize_text(content_response)
            
            # Valida e corrige o JSON
            parsed_response = validate_and_fix_json(cleaned_content)
            
            # Verifica se tem a estrutura esperada
            if "responses" not in parsed_response:
                raise ValueError("Response missing 'responses' field")
            
            responses = parsed_response["responses"]
            if not isinstance(responses, list):
                raise ValueError("'responses' field is not a list")
            
            return responses
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries:
                print(f"All {max_retries + 1} attempts failed")
                return []
            
            # Aguarda um pouco antes de tentar novamente
            import time
            time.sleep(1)
    
    return []

def build_example(title: str, content: str) -> Dict:
    """Cria a estrutura final de um produto com perguntas e respostas."""
    
    questions = [template.format(title=title) for template in QUESTION_TEMPLATES]
    user_prompt = user_prompt_text(questions, content)
    try:
        responses = chat_response(user_prompt)
        # Verifica se obteve o número correto de respostas
        if len(responses) != len(questions):
            print(f"Warning: Expected {len(questions)} responses, got {len(responses)}")
            # Preenche respostas faltantes ou remove extras
            while len(responses) < len(questions):
                responses.append("No response generated")
            responses = responses[:len(questions)]
    except Exception as e:
        print(f"Error processing item with title '{title}': {e}")
        responses = [f"Error generating response: {str(e)}"] * len(questions)

    return {
        "title": title,
        "content": content,
        "questions": questions,
        "responses": responses
    }

def create_finetune_dataset(data, max_examples=-1, start_chunk_index=0):
    """
    Create a finetune dataset in JSONL format.
    
    Args:
        data: Dataset to process
        max_examples: Maximum number of examples to process (-1 for all)
        start_chunk_index: Index of chunk to start from (0 to start from beginning)
    """
    examples = []
    chunk_index = start_chunk_index
    total_count = 0
    items_to_skip = start_chunk_index * CHUNK_SIZE

    print(f"Starting from chunk {start_chunk_index} (skipping first {items_to_skip} valid items)")

    valid_items_processed = 0
    chunk_start_time = time.time()
    
    for item in data:
        title = item.get("title", "")
        content = item.get("content", "")

        if not title or not content or content.strip() == "":
            continue  # ignora produtos inválidos
        
        # Pula itens válidos até chegar no ponto de início
        if valid_items_processed < items_to_skip:
            valid_items_processed += 1
            continue
        
        title = clean_content(title)
        content = clean_content(content)
        example = build_example(title, content)        
        
        examples.append(example)
        total_count += 1
        valid_items_processed += 1

        # Se atingiu o tamanho do chunk, salva e reinicia
        if len(examples) >= CHUNK_SIZE:
            save_chunk(examples, chunk_index)
            chunk_time = time.time() - chunk_start_time
            print(f"Saved chunk {chunk_index} with {len(examples)} items (Time: {chunk_time:.2f}s)")
            chunk_index += 1
            examples = []
            chunk_start_time = time.time()  # Reinicia o timer para o próximo chunk

        if max_examples > 0 and total_count >= max_examples:
            break

    # Salva o último chunk se restarem produtos
    if examples:
        save_chunk(examples, chunk_index)
        chunk_time = time.time() - chunk_start_time
        print(f"Saved final chunk {chunk_index} with {len(examples)} items (Time: {chunk_time:.2f}s)")

    chunks_created = (chunk_index - start_chunk_index + (1 if examples else 0))
    print(f"\nFinished! {total_count} products processed into {chunks_created} chunk(s).")


# CHUNK_SIZE = 25
# MAX_TOTAL_EXAMPLES = 100
data = load_amazon_trn()

create_finetune_dataset(data=data, max_examples=MAX_TOTAL_EXAMPLES, start_chunk_index=3)
# source env/bin/activate
# time python finetune_dataset_chunked.py
# cat finetune_chunks/* > dataset/finetune_dataset_merged.jsonl
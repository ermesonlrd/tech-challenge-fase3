import json

INPUT_FILE = "dataset/finetune_dataset_merged.jsonl"
OUTPUT_FILE = "dataset/instructions_finetune_dataset.jsonl"

def todos_os_itens_sao_strings(lista):
    if not lista:
        return False
    return all(isinstance(item, str) for item in lista)

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        if not line.strip():
            continue
        example = json.loads(line)

        questions = example.get("questions", [])
        responses = example.get("responses", [])
        if not todos_os_itens_sao_strings(responses):
            continue

        # Garante que tenha o mesmo tamanho
        for q, r in zip(questions, responses):
            item = {
                "text": (
                    f"<|system|>\nYou are a helpful assistant.\n"
                    f"<|user|>\n{q.strip()}\n"
                    f"<|assistant|>\n{r.strip()}"
                )
            }
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")


print(f"✅ Dataset pronto para o fine-tuning salvo em: {OUTPUT_FILE}")

# source env/bin/activate
# python instructions_finetune_dataset.py
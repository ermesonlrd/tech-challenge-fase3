# Tech Challenge Fase 3
Projeto de solução ao [desafio proposto](DESAFIO.md), fine-tuning de um foundation model.

## Ambiente e dependências

```bash
sudo apt install python3-venv
python3 -m venv env
source env/bin/activate
```
## Arquivos

* [config.py](config.py) - carrega variáveis de ambiente de um arquivo .env para configurar a URL base, chave da API e o modelo LLM como o ChatGPT, Deepseek ou Ollama através de uma interface OpenAI-compatível.
* [dataset_utils.py](dataset_utils.py) - fornece utilitários para processar o dataset Amazon LF-1.3M, com os métodos: load_amazon_trn() (carrega dados JSON), clean_content() (limpa e trunca conteúdo), save_chunk() (salva blocos em arquivos JSONL), e sanitize_text() (remove caracteres problemáticos e normaliza texto).
* [finetune_dataset_chunked.py](finetune_dataset_chunked.py) - processa o dataset Amazon para criar dados de fine-tuning usando LLMs, com os métodos: validate_and_fix_json() (corrige JSON malformado), chat_response() (chama API com retry e tratamento de erros), build_example() (gera perguntas/respostas para produtos), e create_finetune_dataset() (processa dataset em chunks salvando arquivos JSONL para treinamento).
* [instructions_finetune_dataset.py](instructions_finetune_dataset.py) - converte o dataset de fine-tuning para formato de instruções, com o método: normalize_response() (converte dicts/listas em strings limpas) e processa cada par pergunta-resposta transformando-os no formato adequado para treinamento do modelo.
* [fine-tunning.ipynb](fine-tunning.ipynb) - arquivo do Jupyter Notebook para fine-tuning de LLMs usando Unsloth, contendo: instalação de dependências (unsloth, xformers, bitsandbytes), carregamento do modelo Llama-3-8b com quantização 4-bit, configuração LoRA/PEFT, carregamento do dataset de instruções, configuração do SFTTrainer com parâmetros otimizados, e execução do treinamento.

## Execução
Executar todos comandos no diretório atual.

Configurar as variáveis conforme o modelo escolhido:
```bash
cp .env.template .env
```
### Preparação do dataset
Modelo Utilizado na geração do dataset: llama3.2:3b  
Tempo necessário: por volta de 14 dias.  
Três etapas: 

1) Geração dos arquivos com perguntas e respostas sobre cada produto.  
2) Concatencar os arquivos gerados em um único arquivo.  
3) Converter o arquivo para um formato adequado para treinamento.

#### 1 Geração dos arquivos com perguntas e respostas sobre cada produto
O dataset foi gerado localmente com Ollama utilizando o modelo llama3.2:3b. Ele recebe como entrada o arquivo **./dataset/LF-Amazon-1.3M/trn.json** e gera os arquivos na pasta **./finetune_chunks**. Basicamente são acrescentadas 2 perguntas e 2 respostas para cada produto, basedo no título e no conteúdo.

Os arquivos são gerados com os comandos abaixo:
```bash
source env/bin/activate 
python finetune_dataset_chunked.py
```

Os arquivos serão gerados no formato  finetune_chunks/chunk_{index}.jsonl e, caso algum erro ocorra na geração, é possível continuar a partir do último arquivo gerado. Alterando o parâmetro **start_chunk_index** no arquivo **finetune_dataset_chunked.py**:

```python
create_finetune_dataset(data=data, max_examples=MAX_TOTAL_EXAMPLES, start_chunk_index=99)
```

Formato de arquivos gerados:
```json
  {"title": "Girls Ballet Tutu Neon Pink", "content": "High quality 3 layer ballet tutu. 12 inches in length", "questions": ["Can you describe the product 'Girls Ballet Tutu Neon Pink'?", "What are the main features and advantages of 'Girls Ballet Tutu Neon Pink'?"], "responses": ["The Girls Ballet Tutu Neon Pink is a 3-layer high-quality tutu with a neon pink color, measuring 12 inches in length.", "Some of the main features and advantages of the Girls Ballet Tutu Neon Pink include its high quality construction, neon pink color, and 12-inch length."]}  
```

#### 2 Concatencar os arquivos gerados em um único arquivo
Concatenar os arquivos em um único arquivo chamado **./dataset/finetune_dataset_merged.jsonl**
Os arquivos são concatenados em um único arquivo chamado **./dataset/finetune_dataset_merged.jsonl**.
```bash
cat finetune_chunks/* > dataset/finetune_dataset_merged.jsonl
tail -n 10 dataset/finetune_dataset_merged.jsonl
```

#### 3 Converter o arquivo para um formato adequado para treinamento
Converter o arquivo **./dataset/finetune_dataset_merged.jsonl** para o arquivo **./dataset/instructions_finetune_dataset.jsonl**, pronto para o treinamento do modelo.
```bash
python instructions_finetune_dataset.py
wc -l dataset/instructions_finetune_dataset.jsonl
```

Formato de arquivo gerado:
```json
{"text": "<|system|>\nYou are a helpful assistant.\n<|user|>\nCan you describe the product 'Girls Ballet Tutu Neon Pink'?\n<|assistant|>\nThe Girls Ballet Tutu Neon Pink is a 3-layer high-quality tutu with a neon pink color, measuring 12 inches in length."}
{"text": "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat are the main features and advantages of 'Girls Ballet Tutu Neon Pink'?\n<|assistant|>\nSome of the main features and advantages of the Girls Ballet Tutu Neon Pink include its high quality construction, neon pink color, and 12-inch length."}
```  

### Fine-tuning do modelo
Executar o arquivo **fine-tunning.ipynb** para realizar o treinamento. Recomendado executar ele no Google Colab devido aos recursos de Hardware necessários.

Modelo utilizado para o fine-tuning: unsloth/llama-3-8b-bnb-4bit
Ambiente de treinamento: Google Colab com GPU A100.

Tempo de treinamento:  
1k de dataset e 5 épocas - 08:11  
18k de dataset e 4 épocas - 1:09:30  
187k de dataset e 2 épocas - 5:59:21  

Documentação dos parâmetros estão no arquivo **fine-tunning.ipynb**.

## Conclusão

O projeto mostrou de forma prática como preparar e treinar um modelo de linguagem em larga escala. A preparação do dataset localmente com um modelo menor (llama3.2:3b via Ollama) foi viável, mas apresentou falhas em algumas perguntas e respostas geradas, exigindo cuidados extras na validação. Para datasets muito grandes, o custo de tempo e recursos se torna elevado, o que reforça a importância de estratégias de amostragem e otimização.

O processo de geração, concatenação, conversão e treinamento demonstrou-se estruturado e reprodutível, embora dependente de boa infraestrutura de GPU. Em síntese, a atividade evidenciou tanto o potencial quanto os desafios do fine-tuning, destacando que a qualidade do dataset é tão importante quanto a configuração do treinamento para alcançar bons resultados.
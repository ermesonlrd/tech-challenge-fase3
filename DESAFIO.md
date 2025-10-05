# Tech Challenge Fase 3

## Definição do desafio
### O PROBLEMA
No Tech Challenge desta fase, você precisa executar o fine-tuning de um foundation model (Llama, BERT, MISTRAL etc.), utilizando o dataset "The AmazonTitles-1.3MM". O modelo treinado deverá:

* Receber perguntas com um contexto obtido por meio do arquivo json “trn.json” que está contido dentro do dataset.
* A partir do prompt formado pela pergunta do usuário sobre o título do produto, o modelo deverá gerar uma resposta baseada na pergunta do usuário trazendo como resultado do aprendizado do fine-tuning os dados da sua descrição.

### Fluxo de Trabalho Atualizado:

1. Escolha do Dataset:  
Descrição: O The AmazonTitles-1.3MM consiste em consultas textuais reais de usuários e títulos associados de produtos relevantes encontrados na Amazon e suas descrições, medidos por ações implícitas ou explícitas dos usuários.

2. Preparação do Dataset:  
Faça o download do dataset AmazonTitles-1.3MM (https://drive.google.com/file/d/12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK/view) e utilize o arquivo “trn.json”. Nele, você utilizará as colunas “title” e “content”, que contém título e descrição respectivamente. Prepare os prompts para o fine-tuning, garantindo que estejam organizados de maneira adequada para o treinamento do modelo escolhido. Limpe e pré-processe os dados conforme necessário para o modelo escolhido.

3. Chamada do Foundation Model  
Importe o foundation model que será utilizado e faça um teste apresentando o resultado atual do modelo antes do treinamento. Assim, você terá uma base de análise após o fine-tuning e será possível avaliar a diferença do resultado gerado. 

4. Execução do Fine-Tuning:  
Execute o fine-tuning do foundation model selecionado (por exemplo, BERT, GPT, Llama...) utilizando o dataset preparado. Documente o processo de fine-tuning, incluindo os parâmetros utilizados e qualquer ajuste específico realizado no modelo.

5. Geração de Respostas:  
Configure o modelo treinado para receber perguntas dos usuários. O modelo deverá gerar uma resposta baseada na pergunta do usuário e nos dados provenientes do fine-tuning, incluindo as fontes fornecidas.

O que esperamos para o entregável?

* Documento detalhando o processo de seleção e preparação do dataset.
* Descrição do processo de fine-tuning do modelo, com detalhes dos parâmetros e ajustes utilizados, e o código-fonte do processo de fine-
tuning.
* Um vídeo demonstrando o modelo treinado gerando respostas a partir de perguntas do usuário e utilizando o contexto obtido por meio treinamento com o fine-tuning.

## Considerações para entrega:
* O vídeo deve ter no máximo 10 minutos.
* Os vídeos devem ser enviados para o Youtube, e os códigos devem ser disponibilizados no Github (ou equivalente).
* A entrega deve ser feita em um arquivo PDF, contendo o link do vídeo no YouTube e o repositório do Github com o código denominado “Tech Challenge”.
* Não é preciso apresentar quem é o grupo no início do vídeo, apenas coloque os integrantes do grupo na descrição do vídeo.
* Não será possível entregar o projeto com o mesmo tema apresentado em aula.
* Não é preciso entregar um relatório PDF e nem uma apresentação de slides.
* Foque em gravar um vídeo de apresentação claro e objetivo.

Se você ficou com alguma dúvida, não deixe de acessar o Discord para
que alguém da equipe possa te ajudar.
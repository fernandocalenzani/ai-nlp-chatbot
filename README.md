# Deep Learning e Processamento de Linguagem Natural

Bem-vindo ao meu repositório de estudo sobre Deep Learning e Processamento de Linguagem Natural (D-NLP). Abaixo estão os principais tópicos que serão abordados:

## 2. NLP - Intuicao

1. **Bag-of-Words**

```
  - Insert words in a bag
  - a ordem da frase importa e gera respostas diferentes
  - tamanho das saidas e fixo
  - bag -> bag_of_words = [SOS, EOS, word1, word2, ..., wordn, Special-words] - SOS (start of sentence), EOS (end of sentence)

  EX: Ola, Jones, voce ja voltou para o BR? Gostaria de saber se voce esta por ai. ate, B

    bag_of_words = [1, 1, 0, 0, ..., 3]

    SOS = 1 porque inicia uma vez
    EOS = 1 porque inicia uma vez
    special = Jones, BR, B = 3
```

2. **Seq2Seq**

```
- Arch - [One-to-One, one-to-many, many-to-one, many-to-many-unordered, many-to-many-ordered]
- to use in chatbot: many-to-many-unordered

1. **Step-1**: cada palavra tera um numero de identificacao
2. **Step-2**: Treinamento: Codificador > Decodificador
3. **Step-3**: Beam Search Decoding
4. **Step-4**: cada palavra na codificacao recebe um peso
5. **Step-5**: geracao do vetor *Context Vector*


```

### 3 Tópicos Principais

1. **Pré-processamento dos Dados (14 Tópicos)**

   - Detalhes sobre como preparar os dados para treinamento.

2. **Construção do Modelo Seq2Seq (7 Tópicos)**

   - Passos para construir a arquitetura Seq2Seq.

3. **Treinamento do Modelo Seq2Seq (12 Tópicos)**
   - Exploração do processo de treinamento e otimização do modelo Seq2Seq.

### 5 Tópicos Adicionais

1. **Testes do Modelo Seq2Seq**

   - Estratégias para avaliar o desempenho do modelo.

2. **Melhorias e Ajuste dos Parâmetros do Modelo Seq2Seq**

   - Dicas sobre como ajustar e otimizar os parâmetros do modelo.

3. **Outras Implementações de Chatbot**
   - Exploração de diferentes abordagens e implementações de chatbots.

Sinta-se à vontade para explorar os tópicos de acordo com suas necessidades de estudo. Boa jornada no mundo fascinante do Deep Learning e PLN!

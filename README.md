# Assistente Inteligente de Emergências Veiculares

**Global Solution 2026.2 — Generative AI for Engineering (FIAP)**  
**Projeto:** Assistente de orientação para um MVP de assistência veicular ("Uber do Guincho").

## Integrantes

| Nome                       | RM     |
|----------------------------|--------|
| Márcio Hitoshi Tahyra      | 552511 |
| Jéssica Witzler Costacurta | 99068  |

## 1. Objetivo

O projeto utiliza um modelo de linguagem local e uma base de conhecimento com fontes para oferecer orientações iniciais em emergências veiculares. O assistente identifica possíveis problemas, sugere tipos de assistência e apresenta cuidados de segurança. Não substitui diagnóstico mecânico nem serviços de emergência.

**Assuntos cobertos:** bateria e sistema de partida, pneus, superaquecimento, pane seca e segurança em emergências. Perguntas fora da documentação devem receber recusa, em vez de uma resposta sem fundamento na base.

## 2. Arquitetura

```text
Documentos Markdown com fontes
            |
     Chunking + metadados
            |
  Embeddings multilíngues (E5)
            |
Busca híbrida (semântica + IDF)
            |
   Seleção dos chunks relevantes
            |
   Qwen 2.5 3B via Ollama
            |
 Resposta JSON + referências
            |
  API FastAPI / Avaliação local
```

A configuração **RAG + agente V7** adiciona políticas especializadas para casos de segurança e auditoria estrutural das respostas. A implementação local usa Python, `sentence-transformers`, NumPy, Ollama e FastAPI.

**Observação sobre a API:** na versão atual, `app/api.py` chama o fluxo de `app/assistente_rag.py` (RAG simples). O agente V7 é executado por `app/assistente_agente.py` e utilizado na comparação de configurações. Portanto, não se deve apresentar a API como se já utilizasse o agente V7.

## 3. Organização do repositório

```text
assistente-ia-veicular/
├── app/
│   ├── api.py
│   ├── assistente_rag.py
│   ├── assistente_agente.py       
│   ├── avaliar_configuracoes.py  
│   ├── gerar_chunks.py
│   ├── buscar_chunks.py
│   └── ...
├── base/
│   ├── documentos/
│   │   ├── bateria/
│   │   ├── pneus/
│   │   ├── superaquecimento/
│   │   ├── combustivel/
│   │   └── seguranca/
│   ├── chunks/
│   │   ├── chunks.json
│   │   ├── indice_embeddings.json
│   │   └── embeddings.npy
│   └── metadados/
├── avaliacao/
│   ├── perguntas.json
│   ├── gabarito.json
│   ├── resultados_busca.json
│   ├── respostas_qwen_puro.json
│   ├── respostas_rag_simples.json
│   ├── respostas_rag_agente_v7.json
│   ├── placar_configuracoes.json
│   └── placar_configuracoes.csv
├── notebooks/
├── relatorio/
├── requirements.txt
├── .env.example
└── README.md
```

Alguns arquivos são **gerados na execução** e precisam estar presentes para reproduzir os resultados; confira a pasta `base/chunks/` antes de executar a API ou a avaliação.

## 4. Requisitos

- Windows com PowerShell (ambiente utilizado no desenvolvimento), ou sistema equivalente.
- Python **3.10**.
- [Ollama](https://ollama.com/) instalado e em execução.
- Modelo `qwen2.5:3b` disponível no Ollama.
- Conexão com a internet na primeira instalação para baixar dependências e o modelo de embeddings `intfloat/multilingual-e5-small`.
- GPU é opcional; a execução em CPU pode ser mais lenta. Os testes de desenvolvimento foram feitos com uma RTX 2060 de 6 GB.

### 4.1. Instalação

Abra o terminal na raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

O `requirements.txt` deve conter, no mínimo, as dependências utilizadas: `ollama`, `python-dotenv`, `requests`, `sentence-transformers`, `numpy`, `fastapi` e `uvicorn[standard]`.

Baixe o modelo local:

```powershell
ollama pull qwen2.5:3b
ollama list
```

Mantenha o Ollama aberto. Se o serviço não estiver em execução, inicie-o conforme a instalação do seu sistema.

### 4.2. Variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

Não publique arquivos `.env` com credenciais. Inclua apenas um `.env.example` sem segredos no GitHub.

## 5. Execução do assistente

### 5.1. Testar o RAG simples no terminal

```powershell
python app/assistente_rag.py
```

### 5.2. Testar o agente V7 no terminal

```powershell
python app/assistente_agente.py
```

Faça perguntas sobre situações documentadas, por exemplo:

- `Meu carro não liga e as luzes do painel estão fracas.`
- `É seguro abrir o reservatório de arrefecimento com o motor quente?`
- `Onde devo esperar por ajuda depois de parar o veículo?`

Para testar uma recusa fora da base:

- `A luz do ABS acendeu. Qual peça devo substituir?`

### 5.3. Iniciar a API

```powershell
python -m uvicorn app.api:app --reload
```

Acesse:

- Documentação interativa: <http://127.0.0.1:8000/docs>
- Verificação de saúde: <http://127.0.0.1:8000/saude>
- Endpoint de análise: `POST http://127.0.0.1:8000/analisar`

No Swagger (`/docs`), abra `POST /analisar`, clique em **Try it out** e envie:

```json
{
  "relato": "Meu carro não liga e as luzes do painel estão fracas."
}
```

A API devolve JSON com os campos `status`, `resposta`, `possivel_problema`, `servico_recomendado`, `orientacoes_seguranca`, `pergunta_esclarecimento`, `fontes` e `chunks_utilizados`.

Os valores possíveis de `status` são `respondido`, `esclarecer` e `recusa`. O conteúdo da resposta depende da consulta e das evidências recuperadas.

## 6. Base de conhecimento e recuperação

A base possui **cinco documentos temáticos** e **19 chunks** identificados, cada um com texto e metadados de origem. O embedding utilizado é `intfloat/multilingual-e5-small` (384 dimensões).

A busca combina similaridade semântica (**peso 0,65**) e busca lexical com IDF (**peso 0,35**), selecionando inicialmente até três chunks. O agente V7 também pode consultar evidências documentais adicionais nas políticas especializadas.

Os arquivos `chunks.json`, `indice_embeddings.json` e `embeddings.npy` devem ser mantidos consistentes. Se os documentos forem modificados, gere novamente os artefatos de indexação antes de comparar os resultados.

### Fontes e direitos de uso

As URLs de referência são preservadas nos metadados e nos chunks. Consulte `base/documentos/` e `base/metadados/` para a lista específica de fontes. **Verifique e documente a licença ou as condições de uso de cada fonte antes da publicação**; a presença de uma URL não significa que seu conteúdo tenha licença aberta.

## 7. Reproduzir a avaliação

Com o Ollama ativo e os arquivos da base disponíveis, execute na raiz:

```powershell
python app/avaliar_configuracoes.py
```

O avaliador processa **20 perguntas**, das quais **15 pertencem aos assuntos documentados** e **5 são deliberadamente fora da base**. As configurações são:

1. **Qwen puro:** modelo sem recuperação documental, solicitado a produzir resposta estruturada.
2. **RAG simples:** busca de evidências e geração de resposta.
3. **RAG + agente V7:** RAG com políticas especializadas e auditoria.

Os arquivos de saída são gravados em `avaliacao/`, incluindo respostas individuais, `placar_configuracoes.json` e `placar_configuracoes.csv`. A execução pode demorar alguns minutos, dependendo do hardware.

### 7.1. Resultados observados

| Métrica | Qwen puro | RAG simples | Agente V7 |
|---|---:|---:|---:|
| Acerto de status | 40% (8/20) | 75% (15/20) | **100% (20/20)** |
| Estrutura JSON válida | 65% | 100% | **100%** |
| Citações rastreáveis | N/A | 100% | **100%** |
| Recusas corretas (5 casos fora da base) | 0% | 100% | **100%** |

**Interpretação:** o acerto de status mede a classificação entre responder, esclarecer e recusar, conforme o gabarito adotado. A validação de citações verifica rastreabilidade/consistência de referências, **não** comprova automaticamente a exatidão semântica de cada orientação. A configuração sem RAG não possui citações documentais, por isso esse campo é N/A.

### 7.2. Métricas de recuperação

| Métrica | Semântica | Híbrida |
|---|---:|---:|
| Recall@1 | 26,11% | 31,67% |
| Recall@3 | 73,89% | **87,78%** |
| Hit@1 | 66,67% | 80,00% |
| Hit@3 | 93,33% | **100,00%** |

Essas métricas são da etapa de **recuperação de chunks**, e não do status final das respostas.

## 8. Cinco falhas reais e correções

| Pergunta | Falha observada no RAG simples | Ajuste realizado no agente |
|---|---|---|
| 1 — Partida e painel fraco | Pediu esclarecimento desnecessário | Política de bateria e recarga |
| 6 — Pneu no acostamento | Pediu esclarecimento desnecessário | Política de segurança para pneus |
| 7 — Temperatura alta | Recusou caso documentado | Política de superaquecimento |
| 9 — Reservatório quente | Pediu esclarecimento desnecessário | Orientação direta de não abrir o reservatório quente |
| 14 — Esperar por ajuda | Recusou caso documentado | Política de espera em local protegido |

Após as correções, os cinco casos receberam `respondido` na execução do agente V7. O desempenho foi medido no mesmo conjunto utilizado para identificar e corrigir os erros; é necessário testar perguntas inéditas para estimar generalização.

## 9. Limitações e segurança

- O assistente **não** substitui profissionais mecânicos, socorro rodoviário ou serviços de emergência.
- A base é limitada aos assuntos documentados; a recusa fora de escopo é um comportamento esperado.
- O resultado de 100% se refere às 20 perguntas do teste, não a qualquer pergunta futura.
- A rastreabilidade de citações não substitui uma revisão humana do conteúdo.
- As políticas especializadas foram refinadas com base em erros do próprio conjunto de avaliação.
- O modelo e os embeddings são executados localmente; não publique relatos pessoais de usuários nos arquivos de avaliação ou logs.


---

**Desenvolvido para fins acadêmicos — FIAP Global Solution 2026.2.**

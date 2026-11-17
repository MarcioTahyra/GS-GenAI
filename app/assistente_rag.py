
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter
from math import log

import numpy as np
import ollama
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# CONFIGURAÇÕES
# --------------------------------------------------

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

ARQUIVO_CHUNKS = RAIZ_PROJETO / "base" / "chunks" / "chunks.json"
ARQUIVO_INDICE = RAIZ_PROJETO / "base" / "chunks" / "indice_embeddings.json"
ARQUIVO_EMBEDDINGS = RAIZ_PROJETO / "base" / "chunks" / "embeddings.npy"

MODELO_EMBEDDING = "intfloat/multilingual-e5-small"
LIMIAR_SEMANTICO = 0.82
LIMIAR_PALAVRAS_CHAVE = 0.15

PESO_SEMANTICO = 0.65
PESO_PALAVRAS_CHAVE = 0.35
TOP_K = 3

TERMOS_POR_CATEGORIA = {
    "bateria": [
        "bateria", "alternador", "partida", "luzes fracas",
        "painel fraco", "nao liga", "nao da partida",
        "motor de partida", "cabo de bateria"
    ],
    "pneus": [
        "pneu", "pneus", "furou", "furado", "estepe",
        "pneu vazio", "pneu murcho", "trocar a roda"
    ],
    "combustivel": [
        "combustivel", "gasolina", "etanol", "diesel",
        "pane seca", "tanque vazio", "sem gasolina",
        "sem combustivel"
    ],
    "superaquecimento": [
        "superaquecimento", "motor quente", "temperatura alta",
        "temperatura subindo", "vapor", "radiador",
        "luz de temperatura", "motor esquentando"
    ],
    "seguranca": [
        "acostamento", "rodovia", "triangulo",
        "pisca alerta", "pisca-alerta", "sinalizacao",
        "local perigoso", "parado na via", "prf",
        "policia rodoviaria federal", "numero de emergencia"
    ]
}

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "para", "por", "com", "sem", "e",
    "ou", "que", "qual", "quais", "como", "quando",
    "onde", "meu", "minha", "seu", "sua", "se", "ao",
    "aos", "é", "ser", "está", "estou", "foi", "tem",
    "ter", "pode", "posso", "devo", "fazer", "fazer",
    "carro", "veículo"
}

load_dotenv(RAIZ_PROJETO / ".env")

OLLAMA_HOST = __import__("os").getenv(
    "OLLAMA_HOST", "http://localhost:11434"
)

OLLAMA_MODEL = __import__("os").getenv(
    "OLLAMA_MODEL", "qwen2.5:3b"
)


# --------------------------------------------------
# CARREGAMENTO DA BASE
# --------------------------------------------------

def carregar_json(caminho):
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    with open(caminho, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def carregar_base():
    chunks = carregar_json(ARQUIVO_CHUNKS)
    indice = carregar_json(ARQUIVO_INDICE)

    if not ARQUIVO_EMBEDDINGS.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_EMBEDDINGS}"
        )

    embeddings = np.load(ARQUIVO_EMBEDDINGS)

    metadados_por_id = {
        item["id"]: item.get("metadados", {})
        for item in indice
    }

    for chunk in chunks:
        metadados_indice = metadados_por_id.get(
            chunk["id"], {}
        )

        if not chunk.get("metadados"):
            chunk["metadados"] = metadados_indice

    if len(chunks) != len(embeddings):
        raise ValueError(
            "A quantidade de chunks é diferente da quantidade "
            "de embeddings. Gere novamente os embeddings."
        )

    return chunks, embeddings


# --------------------------------------------------
# PREPARAÇÃO DA BUSCA HÍBRIDA
# --------------------------------------------------

def normalizar_texto(texto):
    texto = texto.lower()

    texto = unicodedata.normalize(
        "NFKD", texto
    ).encode("ASCII", "ignore").decode("ASCII")

    palavras = re.findall(r"\b[a-z0-9]+\b", texto)

    return [
        palavra for palavra in palavras
        if palavra not in STOPWORDS and len(palavra) > 2
    ]


def calcular_idf(chunks):
    documentos = [
        set(normalizar_texto(chunk["texto"]))
        for chunk in chunks
    ]

    total_documentos = len(documentos)
    frequencias = Counter()

    for documento in documentos:
        frequencias.update(documento)

    idf = {}

    for palavra, frequencia in frequencias.items():
        idf[palavra] = log(
            (total_documentos + 1) / (frequencia + 1)
        ) + 1

    return documentos, idf


def calcular_similaridade_palavras(
    consulta, documentos, idf
):
    palavras_consulta = set(normalizar_texto(consulta))

    if not palavras_consulta:
        return np.zeros(len(documentos))

    pesos_consulta = {
        palavra: idf.get(palavra, 1.0)
        for palavra in palavras_consulta
    }

    pontuacoes = []

    for documento in documentos:
        intersecao = palavras_consulta.intersection(
            documento
        )

        if not intersecao:
            pontuacoes.append(0.0)
            continue

        pontuacao = sum(
            pesos_consulta[palavra]
            for palavra in intersecao
        )

        total = sum(pesos_consulta.values())

        pontuacoes.append(
            pontuacao / total if total else 0.0
        )

    return np.array(pontuacoes, dtype=float)


def normalizar_pontuacoes(pontuacoes):
    minimo = np.min(pontuacoes)
    maximo = np.max(pontuacoes)

    if maximo == minimo:
        return np.zeros_like(pontuacoes)

    return (pontuacoes - minimo) / (maximo - minimo)


# --------------------------------------------------
# VERIFICAÇÃO DE ASSUNTOS COBERTOS
# --------------------------------------------------

def normalizar_frase(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(
        caractere for caractere in texto
        if not unicodedata.combining(caractere)
    )
    return re.sub(r"[^a-z0-9]+", " ", texto).strip()


def categorias_da_consulta(consulta):
    texto = f" {normalizar_frase(consulta)} "
    categorias = set()

    for categoria, termos in TERMOS_POR_CATEGORIA.items():
        for termo in termos:
            termo_normalizado = normalizar_frase(termo)
            if f" {termo_normalizado} " in texto:
                categorias.add(categoria)
                break

    return categorias


def selecionar_contexto_compativel(consulta, resultados):
    categorias = categorias_da_consulta(consulta)
    if not categorias:
        return []

    categorias_principais = categorias - {"seguranca"}
    if categorias_principais:
        pertinentes = [
            item for item in resultados
            if item["categoria"] in categorias_principais
        ]
        return pertinentes + [
            item for item in resultados
            if item["categoria"] == "seguranca"
            and item not in pertinentes
        ]

    return [
        item for item in resultados
        if item["categoria"] == "seguranca"
    ]


# --------------------------------------------------
# RECUPERAÇÃO DOS CHUNKS
# --------------------------------------------------

def buscar_contexto(
    consulta,
    chunks,
    embeddings,
    modelo_embedding,
    documentos,
    idf
):
    vetor_consulta = modelo_embedding.encode(
        "query: " + consulta,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings_normalizados = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12
    )

    similaridades_semanticas = (
        embeddings_normalizados @ vetor_consulta
    )

    similaridades_palavras = (
        calcular_similaridade_palavras(
            consulta, documentos, idf
        )
    )

    semantica_normalizada = normalizar_pontuacoes(
        similaridades_semanticas
    )

    palavras_normalizadas = normalizar_pontuacoes(
        similaridades_palavras
    )

    pontuacoes_hibridas = (
        PESO_SEMANTICO * semantica_normalizada
        + PESO_PALAVRAS_CHAVE * palavras_normalizadas
    )

    melhores_indices = np.argsort(
        pontuacoes_hibridas
    )[::-1][:TOP_K]

    resultados = []

    for indice in melhores_indices:
        chunk = chunks[indice]
        metadados = chunk.get("metadados", {})

        resultados.append({
            "id": chunk["id"],
            "texto": chunk["texto"],
            "categoria": metadados.get(
                "categoria", "não informada"
            ),
            "documento": metadados.get(
                "documento", "não informado"
            ),
            "fontes": metadados.get("fontes", []),
            "similaridade_semantica": float(
                similaridades_semanticas[indice]
            ),
            "pontuacao_hibrida": float(
                pontuacoes_hibridas[indice]
            )
        })

    maior_similaridade = float(
        np.max(similaridades_semanticas)
    )
    maior_similaridade_palavras = float(
        np.max(similaridades_palavras)
    )

    return resultados, maior_similaridade, maior_similaridade_palavras


# --------------------------------------------------
# FORMATAÇÃO DO CONTEXTO
# --------------------------------------------------

def montar_contexto(resultados):
    partes = []

    for resultado in resultados:
        fontes = resultado["fontes"]

        fontes_texto = (
            "\n".join(fontes)
            if fontes
            else "Fonte não informada"
        )

        parte = f"""
ID DO CHUNK: {resultado["id"]}
CATEGORIA: {resultado["categoria"]}
DOCUMENTO: {resultado["documento"]}
CONTEÚDO:
{resultado["texto"]}

FONTES:
{fontes_texto}
"""
        partes.append(parte)

    return "\n\n".join(partes)


# --------------------------------------------------
# RESPOSTA DE RECUSA
# --------------------------------------------------

def criar_recusa(motivo):
    return {
        "status": "recusa",
        "resposta": motivo,
        "possivel_problema": None,
        "servico_recomendado": None,
        "orientacoes_seguranca": [],
        "pergunta_esclarecimento": (
            "Você pode fornecer mais detalhes sobre o que aconteceu?"
        ),
        "fontes": [],
        "chunks_utilizados": []
    }


# --------------------------------------------------
# GERAÇÃO COM O OLLAMA
# --------------------------------------------------

def gerar_resposta(
    consulta,
    resultados,
    maior_similaridade,
    maior_similaridade_palavras,
    cliente
):
    resultados = selecionar_contexto_compativel(consulta, resultados)

    if not resultados:
        return criar_recusa(
            "O relato não corresponde a um assunto documentado "
            "na base de conhecimento. Não posso orientar sobre "
            "esse problema sem evidências adequadas."
        )

    if (
        max(item["similaridade_semantica"] for item in resultados)
        < LIMIAR_SEMANTICO
        or maior_similaridade_palavras < LIMIAR_PALAVRAS_CHAVE
    ):
        return criar_recusa(
            "Não encontrei evidências suficientemente relacionadas "
            "ao relato na base de conhecimento."
        )

    contexto = montar_contexto(resultados)

    prompt = f"""
Você é um assistente inteligente de emergências veiculares.
Sua função é interpretar relatos de motoristas e fornecer
orientações de segurança com base EXCLUSIVAMENTE no contexto.

RELATO DO MOTORISTA:
{consulta}

BASE DE CONHECIMENTO:
{contexto}

REGRAS OBRIGATÓRIAS:

1. Use somente informações presentes na base de conhecimento.
2. Não invente diagnósticos. Apresente possibilidades, nunca certezas
   sem evidências.
3. Não invente preços, prazos, disponibilidade ou tempo de chegada.
4. Não afirme que um serviço resolverá o problema sem evidências.
5. Priorize a segurança do motorista e dos passageiros.
6. Se faltarem informações importantes, faça uma pergunta objetiva.
7. Se a base não tiver informação suficiente, use status "recusa".
8. Não recomende reparos em locais perigosos.
9. Inclua somente fontes e IDs de chunks realmente fornecidos.
10. Não inclua informações externas ao contexto.

Retorne SOMENTE um JSON válido, sem Markdown, usando exatamente
esta estrutura:

{{
  "status": "respondido | esclarecer | recusa",
  "resposta": "Orientação em linguagem simples",
  "possivel_problema": "Possível problema ou null",
  "servico_recomendado": "Serviço relacionado ou null",
  "orientacoes_seguranca": [
    "Orientação de segurança"
  ],
  "pergunta_esclarecimento": "Pergunta ou null",
  "fontes": [
    "URL da fonte utilizada"
  ],
  "chunks_utilizados": [
    "id_do_chunk"
  ]
}}

Use:
- "respondido" quando houver evidências suficientes.
- "esclarecer" quando precisar de mais informações.
- "recusa" quando não houver suporte suficiente na base.

Não escolha um serviço como decisão de contratação ou despacho.
Apenas indique o serviço relacionado ao problema possível.
"""

    resposta = cliente.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Responda somente com JSON válido. "
                    "Não invente informações."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0.1,
            "num_ctx": 4096
        }
    )

    conteudo = resposta["message"]["content"]

    try:
        resultado = json.loads(conteudo)
    except json.JSONDecodeError:
        return criar_recusa(
            "Não foi possível gerar uma resposta estruturada "
            "com segurança. Tente reformular o relato."
        )

    if not isinstance(resultado, dict):
        return criar_recusa("O modelo retornou JSON inválido.")

    campos_obrigatorios = [
        "status",
        "resposta",
        "possivel_problema",
        "servico_recomendado",
        "orientacoes_seguranca",
        "pergunta_esclarecimento",
        "fontes",
        "chunks_utilizados"
    ]

    for campo in campos_obrigatorios:
        if campo not in resultado:
            return criar_recusa(
                "A resposta gerada não apresentou todos os campos "
                "necessários. Tente novamente."
            )

    if resultado["status"] not in [
        "respondido", "esclarecer", "recusa"
    ]:
        return criar_recusa(
            "O modelo retornou um status inválido."
        )

    if not isinstance(resultado, dict):
        return criar_recusa("O modelo retornou uma estrutura inválida.")

    for campo in (
        "possivel_problema",
        "servico_recomendado",
        "pergunta_esclarecimento"
    ):
        if isinstance(resultado.get(campo), str):
            if resultado[campo].strip().lower() in {"null", "none", ""}:
                resultado[campo] = None

    if not isinstance(resultado.get("resposta"), str):
        return criar_recusa("O modelo retornou uma resposta inválida.")

    if not isinstance(resultado.get("orientacoes_seguranca"), list):
        return criar_recusa("O modelo retornou orientações inválidas.")

    if not isinstance(resultado.get("chunks_utilizados"), list):
        return criar_recusa("O modelo retornou citações inválidas.")

    if resultado["status"] == "recusa":
        return criar_recusa(resultado["resposta"])

    ids_validos = {item["id"] for item in resultados}
    ids_citados = [
        item for item in resultado["chunks_utilizados"]
        if isinstance(item, str) and item in ids_validos
    ]

    if not ids_citados:
        return criar_recusa(
            "A resposta não apresentou evidências rastreáveis "
            "na base de conhecimento."
        )

    fontes_por_id = {
        item["id"]: item.get("fontes", [])
        for item in resultados
    }
    fontes = []
    for chunk_id in ids_citados:
        for url in fontes_por_id[chunk_id]:
            if url not in fontes:
                fontes.append(url)

    resultado["chunks_utilizados"] = list(dict.fromkeys(ids_citados))
    resultado["fontes"] = fontes

    return resultado


# --------------------------------------------------
# EXECUÇÃO PRINCIPAL
# --------------------------------------------------

def main():
    print("=" * 60)
    print("ASSISTENTE INTELIGENTE DE EMERGÊNCIAS VEICULARES")
    print("=" * 60)

    print("\nCarregando base de conhecimento...")

    chunks, embeddings = carregar_base()

    print(f"{len(chunks)} chunks carregados.")

    print("\nCarregando modelo de embeddings...")

    modelo_embedding = SentenceTransformer(
        MODELO_EMBEDDING
    )

    documentos, idf = calcular_idf(chunks)

    cliente = ollama.Client(host=OLLAMA_HOST)

    print(f"\nModelo de linguagem: {OLLAMA_MODEL}")
    print("Assistente pronto para uso.")
    print("\nDigite 'sair' para encerrar.\n")

    while True:
        consulta = input("Descreva o problema do veículo: ").strip()

        if consulta.lower() in ["sair", "exit", "quit"]:
            print("\nAssistente encerrado.")
            break

        if not consulta:
            print("Digite um relato antes de continuar.\n")
            continue

        try:
            (
                resultados,
                maior_similaridade,
                maior_similaridade_palavras
            ) = buscar_contexto(
                consulta,
                chunks,
                embeddings,
                modelo_embedding,
                documentos,
                idf
            )

            print("\nChunks recuperados:")

            for item in resultados:
                print(
                    f"- {item['id']} "
                    f"(semântica: "
                    f"{item['similaridade_semantica']:.4f})"
                )

            print(
                "\nMaior similaridade semântica: "
                f"{maior_similaridade:.4f}"
            )
            print(
                "Maior correspondência por palavras-chave: "
                f"{maior_similaridade_palavras:.4f}"
            )

            resposta = gerar_resposta(
                consulta,
                resultados,
                maior_similaridade,
                maior_similaridade_palavras,
                cliente
            )

            print("\nRESPOSTA DO ASSISTENTE")
            print("-" * 60)
            print(
                json.dumps(
                    resposta,
                    ensure_ascii=False,
                    indent=4
                )
            )

            print("\n" + "=" * 60 + "\n")

        except Exception as erro:
            print(f"\nErro durante o processamento: {erro}\n")


if __name__ == "__main__":
    main()


import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


RAIZ_PROJETO = Path(__file__).resolve().parent.parent

ARQUIVO_CHUNKS = RAIZ_PROJETO / "base" / "chunks" / "chunks.json"
ARQUIVO_EMBEDDINGS = RAIZ_PROJETO / "base" / "chunks" / "embeddings.npy"

MODELO_EMBEDDINGS = "intfloat/multilingual-e5-small"

PESO_SEMANTICO = 0.65
PESO_PALAVRAS_CHAVE = 0.35

TOP_K = 3

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "no", "na",
    "nos", "nas", "por", "para", "com", "sem", "sobre",
    "que", "e", "ou", "se", "ao", "aos", "à", "às",
    "meu", "minha", "meus", "minhas", "seu", "sua",
    "seus", "suas", "eu", "ele", "ela", "eles", "elas",
    "como", "qual", "quais", "quando", "onde", "devo",
    "posso", "preciso", "está", "estou", "estão",
    "foi", "ser", "ter", "tem", "tenho", "muito",
    "muita", "muitos", "muitas", "mais", "menos",
    "me", "isso", "essa", "esse", "este", "esta"
}


def carregar_json(caminho):
    with open(caminho, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def normalizar_texto(texto):
    texto = str(texto).lower()

    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )

    return texto


def extrair_palavras(texto):
    texto = normalizar_texto(texto)

    palavras = re.findall(r"[a-z0-9]+", texto)

    return [
        palavra
        for palavra in palavras
        if palavra not in STOPWORDS and len(palavra) > 1
    ]


def carregar_dados():
    if not ARQUIVO_CHUNKS.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_CHUNKS}"
        )

    if not ARQUIVO_EMBEDDINGS.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ARQUIVO_EMBEDDINGS}"
        )

    chunks = carregar_json(ARQUIVO_CHUNKS)
    embeddings = np.load(ARQUIVO_EMBEDDINGS)

    if len(chunks) != len(embeddings):
        raise ValueError(
            "A quantidade de chunks não corresponde à quantidade "
            "de embeddings. Execute gerar_chunks.py e atualize "
            "os embeddings antes de continuar."
        )

    return chunks, embeddings


def obter_texto_chunk(chunk):
    return (
        chunk.get("texto")
        or chunk.get("conteudo")
        or chunk.get("text")
        or ""
    )


def obter_id_chunk(chunk, posicao):
    return (
        chunk.get("id")
        or chunk.get("chunk_id")
        or f"chunk_{posicao + 1}"
    )


def calcular_pesos_idf(chunks):
    quantidade_documentos = len(chunks)
    frequencia_documentos = {}

    for chunk in chunks:
        texto = obter_texto_chunk(chunk)
        palavras = set(extrair_palavras(texto))

        for palavra in palavras:
            frequencia_documentos[palavra] = (
                frequencia_documentos.get(palavra, 0) + 1
            )

    idf = {}

    for palavra, frequencia in frequencia_documentos.items():
        idf[palavra] = (
            np.log(
                (quantidade_documentos + 1) /
                (frequencia + 1)
            ) + 1
        )

    return idf


def calcular_similaridade_palavras(
    pergunta,
    chunks,
    idf
):
    palavras_pergunta = extrair_palavras(pergunta)

    if not palavras_pergunta:
        return np.zeros(len(chunks))

    pesos_pergunta = {}

    for palavra in palavras_pergunta:
        pesos_pergunta[palavra] = idf.get(palavra, 1.0)

    peso_total = sum(pesos_pergunta.values())

    resultados = []

    for chunk in chunks:
        texto = obter_texto_chunk(chunk)
        palavras_chunk = set(extrair_palavras(texto))

        peso_encontrado = sum(
            peso
            for palavra, peso in pesos_pergunta.items()
            if palavra in palavras_chunk
        )

        pontuacao = peso_encontrado / peso_total

        resultados.append(pontuacao)

    return np.array(resultados)


def normalizar_scores(scores):
    minimo = np.min(scores)
    maximo = np.max(scores)

    if maximo == minimo:
        return np.ones(len(scores))

    return (scores - minimo) / (maximo - minimo)


def buscar(
    pergunta,
    chunks,
    embeddings,
    modelo,
    idf,
    quantidade=TOP_K
):
    consulta = modelo.encode(
        ["query: " + pergunta],
        normalize_embeddings=True
    )

    consulta = consulta[0]

    embeddings_normalizados = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12
    )

    scores_semanticos = np.dot(
        embeddings_normalizados,
        consulta
    )

    scores_palavras = calcular_similaridade_palavras(
        pergunta,
        chunks,
        idf
    )

    scores_semanticos_normalizados = normalizar_scores(
        scores_semanticos
    )

    scores_hibridos = (
        PESO_SEMANTICO * scores_semanticos_normalizados
        + PESO_PALAVRAS_CHAVE * scores_palavras
    )

    melhores_indices = np.argsort(
        scores_hibridos
    )[::-1][:quantidade]

    resultados = []

    for indice in melhores_indices:
        chunk = chunks[indice]

        resultados.append({
            "id": obter_id_chunk(chunk, indice),
            "categoria": chunk.get("categoria", "Não informada"),
            "documento": chunk.get("documento", "Não informado"),
            "similaridade_semantica": float(
                scores_semanticos[indice]
            ),
            "similaridade_palavras": float(
                scores_palavras[indice]
            ),
            "score_hibrido": float(
                scores_hibridos[indice]
            ),
            "texto": obter_texto_chunk(chunk)
        })

    return resultados


def main():
    print("Carregando dados...")

    chunks, embeddings = carregar_dados()

    print("Carregando modelo de embeddings...")

    modelo = SentenceTransformer(MODELO_EMBEDDINGS)

    idf = calcular_pesos_idf(chunks)

    print(f"\nBase carregada: {len(chunks)} chunks")
    print("Busca híbrida pronta!")
    print("\nDigite uma pergunta ou digite 'sair' para encerrar.")

    while True:
        print("\n" + "=" * 65)

        pergunta = input("Pergunta: ").strip()

        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("Encerrando busca...")
            break

        if not pergunta:
            print("Digite uma pergunta válida.")
            continue

        resultados = buscar(
            pergunta,
            chunks,
            embeddings,
            modelo,
            idf
        )

        print("\nRESULTADOS DA BUSCA HÍBRIDA")
        print("=" * 65)

        for posicao, resultado in enumerate(resultados, start=1):
            print(f"\n{posicao}º lugar")
            print(f"ID: {resultado['id']}")
            print(f"Categoria: {resultado['categoria']}")
            print(f"Documento: {resultado['documento']}")

            print(
                "Similaridade semântica: "
                f"{resultado['similaridade_semantica']:.4f}"
            )

            print(
                "Similaridade por palavras: "
                f"{resultado['similaridade_palavras']:.4f}"
            )

            print(
                "Score híbrido: "
                f"{resultado['score_hibrido']:.4f}"
            )

            print("\nConteúdo:")
            print(resultado["texto"])

            print("-" * 65)


if __name__ == "__main__":
    main()

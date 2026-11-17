
import json
import re
import unicodedata
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


RAIZ_PROJETO = Path(__file__).resolve().parent.parent

ARQUIVO_PERGUNTAS = RAIZ_PROJETO / "avaliacao" / "perguntas.json"
ARQUIVO_GABARITO = RAIZ_PROJETO / "avaliacao" / "gabarito.json"

ARQUIVO_CHUNKS = RAIZ_PROJETO / "base" / "chunks" / "chunks.json"
ARQUIVO_EMBEDDINGS = RAIZ_PROJETO / "base" / "chunks" / "embeddings.npy"
ARQUIVO_INDICE = RAIZ_PROJETO / "base" / "chunks" / "indice_embeddings.json"

ARQUIVO_SAIDA = RAIZ_PROJETO / "avaliacao" / "resultados_busca.json"

MODELO_EMBEDDING = "intfloat/multilingual-e5-small"

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


def carregar_dados():
    arquivos = [
        ARQUIVO_PERGUNTAS,
        ARQUIVO_GABARITO,
        ARQUIVO_CHUNKS,
        ARQUIVO_EMBEDDINGS,
        ARQUIVO_INDICE
    ]

    for caminho in arquivos:
        if not caminho.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {caminho}"
            )

    perguntas = carregar_json(ARQUIVO_PERGUNTAS)
    gabarito = carregar_json(ARQUIVO_GABARITO)
    chunks = carregar_json(ARQUIVO_CHUNKS)
    indice = carregar_json(ARQUIVO_INDICE)
    embeddings = np.load(ARQUIVO_EMBEDDINGS)

    if len(indice) != len(embeddings):
        raise ValueError(
            "Índice e embeddings possuem quantidades diferentes."
        )

    if len(chunks) != len(indice):
        raise ValueError(
            "Chunks e índice possuem quantidades diferentes. "
            "Gere novamente os chunks e os embeddings."
        )

    return perguntas, gabarito, chunks, indice, embeddings


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


def obter_id_chunk(chunk, posicao):
    return (
        chunk.get("id")
        or chunk.get("chunk_id")
        or f"chunk_{posicao + 1}"
    )


def obter_texto_chunk(chunk):
    return (
        chunk.get("texto")
        or chunk.get("conteudo")
        or chunk.get("text")
        or ""
    )


def obter_categoria(item_indice, chunk):
    metadados = item_indice.get("metadados", {})

    return (
        metadados.get("categoria")
        or chunk.get("categoria")
        or "Não informada"
    )


def calcular_pesos_idf(chunks):
    quantidade_documentos = len(chunks)
    frequencia_documentos = {}

    for chunk in chunks:
        palavras = set(
            extrair_palavras(obter_texto_chunk(chunk))
        )

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


def calcular_similaridade_palavras(pergunta, chunks, idf):
    palavras_pergunta = extrair_palavras(pergunta)

    if not palavras_pergunta:
        return np.zeros(len(chunks))

    pesos_pergunta = {}

    for palavra in palavras_pergunta:
        pesos_pergunta[palavra] = idf.get(palavra, 1.0)

    peso_total = sum(pesos_pergunta.values())

    resultados = []

    for chunk in chunks:
        palavras_chunk = set(
            extrair_palavras(obter_texto_chunk(chunk))
        )

        peso_encontrado = sum(
            peso
            for palavra, peso in pesos_pergunta.items()
            if palavra in palavras_chunk
        )

        resultados.append(peso_encontrado / peso_total)

    return np.array(resultados)


def normalizar_scores(scores):
    minimo = np.min(scores)
    maximo = np.max(scores)

    if maximo == minimo:
        return np.ones(len(scores))

    return (scores - minimo) / (maximo - minimo)


def calcular_metricas(recalls_top1, recalls_top3):
    total = len(recalls_top1)

    return {
        "recall_at_1": float(np.mean(recalls_top1)),
        "recall_at_3": float(np.mean(recalls_top3)),
        "hit_rate_at_1": float(
            sum(valor > 0 for valor in recalls_top1) / total
        ),
        "hit_rate_at_3": float(
            sum(valor > 0 for valor in recalls_top3) / total
        )
    }


def avaliar_perguntas(
    modelo,
    perguntas,
    gabarito,
    chunks,
    indice,
    embeddings
):
    resultados = []

    recalls_hibrido_top1 = []
    recalls_hibrido_top3 = []

    recalls_semantico_top1 = []
    recalls_semantico_top3 = []

    acertos_hibrido_top1 = 0
    acertos_hibrido_top3 = 0

    acertos_semantico_top1 = 0
    acertos_semantico_top3 = 0

    mapa_chunks = {}

    for posicao, item in enumerate(indice):
        chunk_id = item["id"]

        mapa_chunks[chunk_id] = {
            "categoria": obter_categoria(item, chunks[posicao]),
            "texto": obter_texto_chunk(chunks[posicao])
        }

    ids_disponiveis = set(mapa_chunks.keys())

    embeddings_normalizados = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12
    )

    idf = calcular_pesos_idf(chunks)

    for item in perguntas:
        identificador = str(item["id"])
        pergunta = item["pergunta"]

        relevantes = gabarito.get(identificador)

        vetor = modelo.encode(
            ["query: " + pergunta],
            normalize_embeddings=True
        )[0]

        scores_semanticos = np.dot(
            embeddings_normalizados,
            vetor
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

        melhores_hibridos = np.argsort(
            scores_hibridos
        )[::-1][:TOP_K]

        melhores_semanticos = np.argsort(
            scores_semanticos
        )[::-1][:TOP_K]

        encontrados_hibridos = []
        encontrados_semanticos = []

        for posicao in melhores_hibridos:
            chunk_id = indice[int(posicao)]["id"]

            encontrados_hibridos.append({
                "id": chunk_id,
                "categoria": mapa_chunks[chunk_id]["categoria"],
                "score_hibrido": float(scores_hibridos[posicao]),
                "similaridade_semantica": float(
                    scores_semanticos[posicao]
                ),
                "similaridade_palavras": float(
                    scores_palavras[posicao]
                )
            })

        for posicao in melhores_semanticos:
            chunk_id = indice[int(posicao)]["id"]

            encontrados_semanticos.append({
                "id": chunk_id,
                "categoria": mapa_chunks[chunk_id]["categoria"],
                "similaridade": float(scores_semanticos[posicao])
            })

        maior_similaridade = float(np.max(scores_semanticos))

        resultado = {
            "id_pergunta": identificador,
            "pergunta": pergunta,
            "maior_similaridade_semantica": maior_similaridade,
            "resultados_hibridos": encontrados_hibridos,
            "resultados_semanticos": encontrados_semanticos
        }

        if relevantes is None:
            resultado["tipo"] = "fora_da_base"

            resultados.append(resultado)

            print(f"\nPergunta {identificador}: {pergunta}")
            print("Tipo: Fora da base")
            print(
                "Top 3 híbrido:",
                [r["id"] for r in encontrados_hibridos]
            )

            continue

        if len(relevantes) == 0:
            raise ValueError(
                f"O gabarito da pergunta {identificador} está vazio."
            )

        ids_invalidos = set(relevantes) - ids_disponiveis

        if ids_invalidos:
            raise ValueError(
                f"IDs inexistentes no gabarito da pergunta "
                f"{identificador}: {ids_invalidos}"
            )

        ids_relevantes = set(relevantes)

        ids_hibrido_top1 = {
            encontrados_hibridos[0]["id"]
        }

        ids_hibrido_top3 = {
            resultado["id"]
            for resultado in encontrados_hibridos
        }

        ids_semantico_top1 = {
            encontrados_semanticos[0]["id"]
        }

        ids_semantico_top3 = {
            resultado["id"]
            for resultado in encontrados_semanticos
        }

        recall_hibrido_1 = (
            len(ids_hibrido_top1 & ids_relevantes)
            / len(ids_relevantes)
        )

        recall_hibrido_3 = (
            len(ids_hibrido_top3 & ids_relevantes)
            / len(ids_relevantes)
        )

        recall_semantico_1 = (
            len(ids_semantico_top1 & ids_relevantes)
            / len(ids_relevantes)
        )

        recall_semantico_3 = (
            len(ids_semantico_top3 & ids_relevantes)
            / len(ids_relevantes)
        )

        recalls_hibrido_top1.append(recall_hibrido_1)
        recalls_hibrido_top3.append(recall_hibrido_3)

        recalls_semantico_top1.append(recall_semantico_1)
        recalls_semantico_top3.append(recall_semantico_3)

        hit_hibrido_1 = recall_hibrido_1 > 0
        hit_hibrido_3 = recall_hibrido_3 > 0

        hit_semantico_1 = recall_semantico_1 > 0
        hit_semantico_3 = recall_semantico_3 > 0

        acertos_hibrido_top1 += hit_hibrido_1
        acertos_hibrido_top3 += hit_hibrido_3

        acertos_semantico_top1 += hit_semantico_1
        acertos_semantico_top3 += hit_semantico_3

        resultado.update({
            "tipo": "com_resposta",
            "chunks_relevantes": relevantes,
            "recall_hibrido_at_1": recall_hibrido_1,
            "recall_hibrido_at_3": recall_hibrido_3,
            "recall_semantico_at_1": recall_semantico_1,
            "recall_semantico_at_3": recall_semantico_3,
            "hit_hibrido_at_1": hit_hibrido_1,
            "hit_hibrido_at_3": hit_hibrido_3,
            "hit_semantico_at_1": hit_semantico_1,
            "hit_semantico_at_3": hit_semantico_3
        })

        resultados.append(resultado)

        print(f"\nPergunta {identificador}: {pergunta}")
        print(f"Maior similaridade semântica: {maior_similaridade:.4f}")
        print(f"Trechos relevantes: {relevantes}")

        print(
            "Top 3 híbrido:",
            [r["id"] for r in encontrados_hibridos]
        )

        print(
            "Top 3 semântico:",
            [r["id"] for r in encontrados_semanticos]
        )

        print(f"Recall híbrido@1: {recall_hibrido_1:.4f}")
        print(f"Recall híbrido@3: {recall_hibrido_3:.4f}")

    total_validas = len(recalls_hibrido_top1)
    total_fora = len(perguntas) - total_validas

    if total_validas == 0:
        raise ValueError(
            "Nenhuma pergunta com gabarito foi encontrada."
        )

    resumo_hibrido = calcular_metricas(
        recalls_hibrido_top1,
        recalls_hibrido_top3
    )

    resumo_semantico = calcular_metricas(
        recalls_semantico_top1,
        recalls_semantico_top3
    )

    print("\n" + "=" * 65)
    print("COMPARAÇÃO DOS MÉTODOS DE BUSCA")
    print("=" * 65)

    print(f"Perguntas com resposta: {total_validas}")
    print(f"Perguntas fora da base: {total_fora}")

    print("\nBUSCA SEMÂNTICA")

    print(
        f"Recall@1: {resumo_semantico['recall_at_1']:.2%}"
    )
    print(
        f"Recall@3: {resumo_semantico['recall_at_3']:.2%}"
    )
    print(
        f"Hit Rate@1: {resumo_semantico['hit_rate_at_1']:.2%}"
    )
    print(
        f"Hit Rate@3: {resumo_semantico['hit_rate_at_3']:.2%}"
    )

    print("\nBUSCA HÍBRIDA")

    print(
        f"Recall@1: {resumo_hibrido['recall_at_1']:.2%}"
    )
    print(
        f"Recall@3: {resumo_hibrido['recall_at_3']:.2%}"
    )
    print(
        f"Hit Rate@1: {resumo_hibrido['hit_rate_at_1']:.2%}"
    )
    print(
        f"Hit Rate@3: {resumo_hibrido['hit_rate_at_3']:.2%}"
    )

    print("\nDIFERENÇA (HÍBRIDA - SEMÂNTICA)")

    for metrica in resumo_hibrido:
        diferenca = (
            resumo_hibrido[metrica]
            - resumo_semantico[metrica]
        )

        print(f"{metrica}: {diferenca:+.2%}")

    return resultados, resumo_hibrido, resumo_semantico


def avaliar_limiares(resultados):
    conhecidos = [
        r["maior_similaridade_semantica"]
        for r in resultados
        if r["tipo"] == "com_resposta"
    ]

    ausentes = [
        r["maior_similaridade_semantica"]
        for r in resultados
        if r["tipo"] == "fora_da_base"
    ]

    if not conhecidos or not ausentes:
        print("\nNão há perguntas suficientes para avaliar o limiar.")
        return None

    candidatos = sorted(set(conhecidos + ausentes))

    limites = [
        (candidatos[i] + candidatos[i + 1]) / 2
        for i in range(len(candidatos) - 1)
    ]

    if not limites:
        print("\nNão foi possível calcular um limiar experimental.")
        return None

    melhores = []

    for limiar in limites:
        verdadeiros_positivos = sum(
            score >= limiar for score in conhecidos
        )

        falsos_negativos = (
            len(conhecidos) - verdadeiros_positivos
        )

        verdadeiros_negativos = sum(
            score < limiar for score in ausentes
        )

        falsos_positivos = (
            len(ausentes) - verdadeiros_negativos
        )

        sensibilidade = (
            verdadeiros_positivos / len(conhecidos)
        )

        especificidade = (
            verdadeiros_negativos / len(ausentes)
        )

        acuracia_balanceada = (
            sensibilidade + especificidade
        ) / 2

        melhores.append({
            "limiar_semantico": float(limiar),
            "sensibilidade": sensibilidade,
            "especificidade": especificidade,
            "acuracia_balanceada": acuracia_balanceada,
            "falsas_rejeicoes": falsos_negativos,
            "falsas_aceitacoes": falsos_positivos
        })

    melhor = max(
        melhores,
        key=lambda resultado: resultado["acuracia_balanceada"]
    )

    print("\n" + "=" * 65)
    print("ANÁLISE DO LIMIAR SEMÂNTICO")
    print("=" * 65)

    print(
        f"Similaridade mínima das perguntas conhecidas: "
        f"{min(conhecidos):.4f}"
    )

    print(
        f"Similaridade máxima das perguntas conhecidas: "
        f"{max(conhecidos):.4f}"
    )

    print(
        f"Similaridade mínima das perguntas fora da base: "
        f"{min(ausentes):.4f}"
    )

    print(
        f"Similaridade máxima das perguntas fora da base: "
        f"{max(ausentes):.4f}"
    )

    print("\nMelhor limiar experimental:")

    print(
        f"Limiar: {melhor['limiar_semantico']:.4f}"
    )

    print(
        f"Sensibilidade: {melhor['sensibilidade']:.2%}"
    )

    print(
        f"Especificidade: {melhor['especificidade']:.2%}"
    )

    print(
        f"Acurácia balanceada: "
        f"{melhor['acuracia_balanceada']:.2%}"
    )

    print(
        f"Falsas rejeições: {melhor['falsas_rejeicoes']}"
    )

    print(
        f"Falsas aceitações: {melhor['falsas_aceitacoes']}"
    )

    if (
        min(ausentes) <= max(conhecidos)
        and max(ausentes) >= min(conhecidos)
    ):
        print(
            "\nATENÇÃO: existe sobreposição entre as perguntas "
            "conhecidas e as perguntas fora da base."
        )

        print(
            "O limiar sozinho não é suficiente para decidir "
            "quando o assistente deve recusar."
        )

    return melhor


def main():
    print("Carregando dados...")

    perguntas, gabarito, chunks, indice, embeddings = (
        carregar_dados()
    )

    print("Carregando modelo de embeddings...")

    modelo = SentenceTransformer(
        MODELO_EMBEDDING,
        device="cpu"
    )

    print("Executando avaliação híbrida...")

    resultados, resumo_hibrido, resumo_semantico = (
        avaliar_perguntas(
            modelo,
            perguntas,
            gabarito,
            chunks,
            indice,
            embeddings
        )
    )

    melhor_limiar = avaliar_limiares(resultados)

    saida = {
        "quantidade_perguntas": len(perguntas),
        "pesos_busca": {
            "semantica": PESO_SEMANTICO,
            "palavras_chave": PESO_PALAVRAS_CHAVE
        },
        "busca_semantica": resumo_semantico,
        "busca_hibrida": resumo_hibrido,
        "melhor_limiar_semantico_experimental": melhor_limiar,
        "resultados_individuais": resultados
    }

    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as arquivo:
        json.dump(
            saida,
            arquivo,
            ensure_ascii=False,
            indent=4
        )

    print(f"\nResultados salvos em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    main()

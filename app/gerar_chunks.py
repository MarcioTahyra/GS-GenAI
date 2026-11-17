
import json
import hashlib
from pathlib import Path


PASTA_DOCUMENTOS = Path("base/documentos")
PASTA_CHUNKS = Path("base/chunks")
PASTA_METADADOS = Path("base/metadados")

TAMANHO_CHUNK = 800
SOBREPOSICAO = 150


def gerar_hash(texto):
    return hashlib.sha256(
        texto.encode("utf-8")
    ).hexdigest()


def dividir_texto(texto):
    palavras = texto.split()

    chunks = []
    inicio = 0

    while inicio < len(palavras):
        fim = inicio
        tamanho_atual = 0

        while fim < len(palavras):
            tamanho_palavra = len(palavras[fim])

            if fim > inicio:
                tamanho_palavra += 1

            if tamanho_atual + tamanho_palavra > TAMANHO_CHUNK:
                break

            tamanho_atual += tamanho_palavra
            fim += 1

        if fim == inicio:
            fim += 1

        conteudo = " ".join(palavras[inicio:fim])
        chunks.append(conteudo)

        if fim >= len(palavras):
            break

        tamanho_sobreposicao = 0
        novo_inicio = fim

        while novo_inicio > inicio:
            palavra = palavras[novo_inicio - 1]

            incremento = len(palavra)

            if novo_inicio < fim:
                incremento += 1

            if tamanho_sobreposicao + incremento > SOBREPOSICAO:
                break

            tamanho_sobreposicao += incremento
            novo_inicio -= 1

        inicio = max(novo_inicio, inicio + 1)

    return chunks


def identificar_categoria(caminho):
    return caminho.parent.name


def extrair_fontes(texto):
    fontes = []

    for linha in texto.splitlines():
        if "http://" in linha or "https://" in linha:
            for palavra in linha.split():
                if palavra.startswith(("http://", "https://")):
                    fontes.append(
                        palavra.rstrip(".,;)")
                    )

    return list(dict.fromkeys(fontes))


def processar_documentos():
    PASTA_CHUNKS.mkdir(parents=True, exist_ok=True)
    PASTA_METADADOS.mkdir(parents=True, exist_ok=True)

    documentos = sorted(
        PASTA_DOCUMENTOS.rglob("*.md")
    )

    if not documentos:
        print("Nenhum documento Markdown foi encontrado.")
        return

    todos_chunks = []
    metadados = []

    for documento in documentos:
        texto = documento.read_text(
            encoding="utf-8"
        ).strip()

        if not texto:
            print(f"Documento vazio ignorado: {documento}")
            continue

        categoria = identificar_categoria(documento)
        nome_documento = documento.stem

        partes = dividir_texto(texto)
        fontes = extrair_fontes(texto)

        print(f"\nDocumento: {documento}")
        print(f"Categoria: {categoria}")
        print(f"Chunks gerados: {len(partes)}")

        for indice, conteudo in enumerate(partes, start=1):
            identificador = f"{nome_documento}_{indice:03d}"

            dados_metadados = {
                "documento": documento.name,
                "caminho_origem": documento.as_posix(),
                "categoria": categoria,
                "numero_chunk": indice,
                "total_chunks_documento": len(partes),
                "tamanho_caracteres": len(conteudo),
                "hash": gerar_hash(conteudo),
                "fontes": fontes
            }

            todos_chunks.append({
                "id": identificador,
                "texto": conteudo,
                "metadados": dados_metadados
            })

            metadados.append({
                "id": identificador,
                **dados_metadados
            })

    arquivo_chunks = PASTA_CHUNKS / "chunks.json"
    arquivo_metadados = PASTA_METADADOS / "metadados.json"

    arquivo_chunks.write_text(
        json.dumps(
            todos_chunks,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    arquivo_metadados.write_text(
        json.dumps(
            metadados,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 50)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 50)
    print(f"Documentos encontrados: {len(documentos)}")
    print(f"Chunks gerados: {len(todos_chunks)}")
    print(f"Arquivo: {arquivo_chunks}")
    print(f"Arquivo: {arquivo_metadados}")


if __name__ == "__main__":
    processar_documentos()

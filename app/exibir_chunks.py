
import json
from pathlib import Path

ARQUIVO = Path("base/chunks/chunks.json")


def main():
    if not ARQUIVO.exists():
        print("Arquivo chunks.json não encontrado.")
        return

    with open(ARQUIVO, "r", encoding="utf-8") as arquivo:
        chunks = json.load(arquivo)

    print("=" * 70)
    print("CONTEÚDO DOS CHUNKS DA BASE DE CONHECIMENTO")
    print("=" * 70)

    for chunk in chunks:
        metadados = chunk["metadados"]

        print("\n" + "=" * 70)
        print(f"ID: {chunk['id']}")
        print(f"Categoria: {metadados['categoria']}")
        print(f"Documento: {metadados['documento']}")
        print("=" * 70)
        print(chunk["texto"])

    print("\nTotal de chunks:", len(chunks))


if __name__ == "__main__":
    main()

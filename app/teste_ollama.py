
import os

import ollama
from dotenv import load_dotenv

load_dotenv()

MODELO = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

cliente = ollama.Client(host=HOST)


def testar_modelo():
    print("=" * 50)
    print("ASSISTENTE INTELIGENTE DE EMERGÊNCIAS VEICULARES")
    print("=" * 50)

    print(f"\nModelo: {MODELO}")
    print(f"Servidor: {HOST}")

    pergunta = (
        "Meu carro não liga e as luzes do painel estão fracas. "
        "Quais podem ser as causas?"
    )

    print("\nPergunta:")
    print(pergunta)

    try:
        resposta = cliente.chat(
            model=MODELO,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente de emergências veiculares. "
                        "Responda em português brasileiro. "
                        "Não invente informações. "
                        "Quando não tiver informações suficientes, "
                        "deixe isso claro. "
                        "Priorize a segurança do motorista."
                    )
                },
                {
                    "role": "user",
                    "content": pergunta
                }
            ],
            stream=False
        )

        print("\nResposta do modelo:")
        print(resposta.message.content)

        print("\nTeste concluído com sucesso!")

    except Exception as erro:
        print("\nErro ao executar o modelo:")
        print(erro)


if __name__ == "__main__":
    testar_modelo()

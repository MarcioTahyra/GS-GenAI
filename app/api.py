from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

try:
    from app.assistente_rag import (
        MODELO_EMBEDDING,
        OLLAMA_HOST,
        OLLAMA_MODEL,
        buscar_contexto,
        calcular_idf,
        carregar_base,
    )
    from app.assistente_agente import gerar_resposta_agente
except ModuleNotFoundError:
    from assistente_rag import (
        MODELO_EMBEDDING,
        OLLAMA_HOST,
        OLLAMA_MODEL,
        buscar_contexto,
        calcular_idf,
        carregar_base,
    )
    from assistente_agente import gerar_resposta_agente


class RelatoEntrada(BaseModel):
    relato: str = Field(min_length=5, max_length=2000, description="Relato do problema do veículo")


class RespostaAnalise(BaseModel):
    status: Literal["respondido", "esclarecer", "recusa"]
    resposta: str
    possivel_problema: str | None = None
    servico_recomendado: str | None = None
    orientacoes_seguranca: list[str]
    pergunta_esclarecimento: str | None = None
    fontes: list[str]
    chunks_utilizados: list[str]


estado = {}
trava_modelo = Lock()


@asynccontextmanager
async def ciclo_vida(app: FastAPI):
    chunks, embeddings = carregar_base()
    modelo_embedding = SentenceTransformer(MODELO_EMBEDDING)
    documentos, idf = calcular_idf(chunks)
    cliente = ollama.Client(host=OLLAMA_HOST, timeout=120)

    estado.update({
        "chunks": chunks,
        "embeddings": embeddings,
        "modelo_embedding": modelo_embedding,
        "documentos": documentos,
        "idf": idf,
        "cliente": cliente,
    })
    try:
        yield
    finally:
        estado.clear()


app = FastAPI(
    title="Assistente Inteligente de Emergências Veiculares",
    description="API local para orientação veicular fundamentada em base de conhecimento.",
    version="1.1.0",
    lifespan=ciclo_vida,
)


@app.get("/")
def inicio():
    return {
        "aplicacao": "Assistente Inteligente de Emergências Veiculares",
        "versao": "1.1.0",
        "documentacao": "/docs",
        "endpoint_analise": "/analisar",
        "configuracao": "rag_agente_v7",
    }


@app.get("/saude")
def saude():
    return {
        "status": "pronto" if estado else "inicializando",
        "modelo": OLLAMA_MODEL,
        "configuracao": "rag_agente_v7",
        "chunks_carregados": len(estado.get("chunks", [])),
    }


def processar_relato(relato: str):
    with trava_modelo:
        resultados, maior_semantica, maior_palavras = buscar_contexto(
            relato,
            estado["chunks"],
            estado["embeddings"],
            estado["modelo_embedding"],
            estado["documentos"],
            estado["idf"],
        )
        return gerar_resposta_agente(
            relato,
            resultados,
            maior_semantica,
            maior_palavras,
            estado["cliente"],
        )


@app.post("/analisar", response_model=RespostaAnalise)
async def analisar(entrada: RelatoEntrada):
    if not estado:
        raise HTTPException(status_code=503, detail="Assistente ainda não está pronto.")

    relato = entrada.relato.strip()
    if len(relato) < 5:
        raise HTTPException(status_code=422, detail="Informe um relato com pelo menos 5 caracteres.")

    try:
        resposta = await run_in_threadpool(processar_relato, relato)
        return RespostaAnalise.model_validate(resposta)
    except ollama.ResponseError as erro:
        raise HTTPException(
            status_code=503,
            detail="O modelo local não está disponível. Verifique o Ollama.",
        ) from erro
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail="Não foi possível analisar o relato neste momento.",
        ) from erro

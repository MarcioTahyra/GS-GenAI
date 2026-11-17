"""Compara Qwen puro, RAG simples e RAG + agente V7.
Execute na raiz do projeto: python app/avaliar_configuracoes.py
Os resultados são medidos localmente; métricas sem gabarito são marcadas N/A.
"""
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import ollama
from sentence_transformers import SentenceTransformer

try:
    from . import assistente_rag as rag
    from . import assistente_agente as agente
except ImportError:
    import assistente_rag as rag
    import assistente_agente as agente

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / 'avaliacao'
CONFIGURACOES = ('qwen_puro', 'rag_simples', 'rag_agente_v7')
CAMPOS = ('status', 'resposta', 'possivel_problema', 'servico_recomendado',
          'orientacoes_seguranca', 'pergunta_esclarecimento', 'fontes', 'chunks_utilizados')


def carregar_json(nome):
    caminho = PASTA / nome
    if not caminho.exists():
        raise FileNotFoundError(f'Arquivo obrigatório não encontrado: {caminho}')
    return json.loads(caminho.read_text(encoding='utf-8-sig'))


def lista_registros(obj, nome):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for chave in ('perguntas', 'questoes', 'casos', 'testes', 'items', 'dados', 'gabarito', 'respostas'):
            if isinstance(obj.get(chave), list):
                return obj[chave]
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return [dict(v, id=k) for k, v in obj.items()]
    raise ValueError(f'Formato de {nome} não reconhecido. Esperado lista ou objeto contendo lista de casos.')


def primeiro(reg, chaves):
    for chave in chaves:
        if chave in reg and reg[chave] is not None:
            return reg[chave]
    return None


def normalizar_id(v):
    return str(v).strip() if v is not None else ''


def carregar_casos():
    perguntas = lista_registros(carregar_json('perguntas.json'), 'perguntas.json')
    bruto = carregar_json('gabarito.json')
    if isinstance(bruto, dict) and all(isinstance(v, list) for v in bruto.values()):
        mapa_gabarito = {
            normalizar_id(chave): {'chunks_relevantes': valor}
            for chave, valor in bruto.items()
        }
    else:
        registros = lista_registros(bruto, 'gabarito.json')
        mapa_gabarito = {}
        for i, reg in enumerate(registros):
            if not isinstance(reg, dict):
                raise ValueError('Cada item do gabarito deve ser um objeto.')
            chave = normalizar_id(primeiro(reg, ('id', 'pergunta_id', 'questao_id', 'numero'))) or str(i + 1)
            if chave in mapa_gabarito:
                raise ValueError(f'ID repetido no gabarito: {chave}')
            mapa_gabarito[chave] = reg

    casos = []
    ids_perguntas = set()
    for i, reg in enumerate(perguntas):
        if isinstance(reg, str):
            reg = {'pergunta': reg}
        if not isinstance(reg, dict):
            raise ValueError(f'Pergunta {i + 1} não é objeto ou texto.')
        pergunta = primeiro(reg, ('pergunta', 'consulta', 'relato', 'texto', 'questao', 'input', 'query'))
        if not isinstance(pergunta, str) or not pergunta.strip():
            raise ValueError(f'Pergunta {i + 1} sem texto.')
        chave = normalizar_id(primeiro(reg, ('id', 'pergunta_id', 'questao_id', 'numero'))) or str(i + 1)
        if chave in ids_perguntas:
            raise ValueError(f'ID de pergunta repetido: {chave}')
        ids_perguntas.add(chave)
        categoria = reg.get('categoria_esperada', '__ausente__')
        esperado = dict(mapa_gabarito.get(chave, {}))
        if categoria is None:
            if chave in mapa_gabarito:
                raise ValueError(f'Pergunta {chave} fora da base possui chunks no gabarito.')
            esperado['status_esperado'] = 'recusa'
        elif isinstance(categoria, str) and categoria.strip():
            if chave not in mapa_gabarito:
                raise ValueError(f'Pergunta {chave} da categoria {categoria} não tem gabarito de chunks.')
            esperado['status_esperado'] = 'respondido'
            esperado['categoria_esperada'] = categoria.strip()
        elif categoria == '__ausente__':
            if chave not in mapa_gabarito and status_esperado(esperado) is None:
                raise ValueError(f'Pergunta {chave} sem gabarito e sem categoria_esperada.')
        else:
            raise ValueError(f'categoria_esperada inválida na pergunta {chave}.')
        casos.append({'id': chave, 'pergunta': pergunta.strip(), 'gabarito': esperado})

    extras = set(mapa_gabarito) - ids_perguntas
    if extras:
        raise ValueError(f'IDs no gabarito sem pergunta correspondente: {sorted(extras)}')
    return casos


def status_esperado(g):
    v = primeiro(g, ('status_esperado', 'status', 'resultado_esperado', 'deve_recusar', 'recusa_esperada', 'fora_da_base', 'nao_consta'))
    if isinstance(v, bool):
        return 'recusa' if v else 'respondido'
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('recusa', 'recusar', 'não consta', 'nao consta', 'fora da base', 'fora_da_base', 'ood', 'não', 'nao'):
            return 'recusa'
        if s in ('respondido', 'responder', 'resposta', 'sim', 'dentro da base', 'in_domain'):
            return 'respondido'
        if s in ('esclarecer', 'esclarecimento'):
            return 'esclarecer'
    return None


def categoria_esperada(g):
    v = primeiro(g, ('categoria_esperada', 'categoria', 'tema', 'classe'))
    return str(v).strip().lower() if isinstance(v, str) else None


def ids_esperados(g):
    v = primeiro(g, ('chunks_relevantes', 'ids_relevantes', 'chunks_esperados', 'chunks', 'ids_chunks'))
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    return None


def qwen_puro(cliente, pergunta):
    """Baseline sem acesso aos documentos, com o mesmo contrato de saída JSON."""
    sistema = (
        'Você é um assistente de emergência veicular. Você NÃO tem acesso à base '
        'de documentos do projeto. Responda em português SOMENTE com um objeto JSON. '
        'Chaves obrigatórias: status, resposta, possivel_problema, '
        'servico_recomendado, orientacoes_seguranca, pergunta_esclarecimento, '
        'fontes, chunks_utilizados. '
        'status deve ser respondido, esclarecer ou recusa. '
        'Use null para campos opcionais desconhecidos e lista de strings para '
        'orientacoes_seguranca. Como não há documentos, fontes e chunks_utilizados '
        'devem ser listas vazias. Não invente preços, prazos, disponibilidade, '
        'diagnóstico confirmado nem a identidade de peças defeituosas. '
        'Se não puder orientar com segurança, use status recusa.'
    )
    resposta = cliente.chat(
        model=rag.OLLAMA_MODEL,
        messages=[{'role': 'system', 'content': sistema},
                  {'role': 'user', 'content': pergunta}],
        format='json', options={'temperature': 0, 'num_ctx': 4096}
    )
    conteudo = resposta['message']['content']
    try:
        return json.loads(conteudo)
    except (ValueError, TypeError):
        return {'status': 'erro_formato', 'resposta': str(conteudo),
                'fontes': [], 'chunks_utilizados': []}


def validar_estrutura(resposta):
    return (isinstance(resposta, dict)
            and resposta.get('status') in ('respondido', 'esclarecer', 'recusa')
            and isinstance(resposta.get('resposta'), str)
            and bool(resposta['resposta'].strip())
            and isinstance(resposta.get('orientacoes_seguranca'), list)
            and isinstance(resposta.get('fontes'), list)
            and isinstance(resposta.get('chunks_utilizados'), list)
            and all(c in resposta for c in CAMPOS))


def avaliar_resposta(resposta, g, chunks_por_id, configuracao):
    if not isinstance(resposta, dict):
        resposta = {}
    esperado = status_esperado(g)
    obtido = resposta.get('status')
    status_correto = (obtido == esperado) if esperado is not None else None
    ids = resposta.get('chunks_utilizados', [])
    fontes = resposta.get('fontes', [])
    ids = ids if isinstance(ids, list) else []
    fontes = fontes if isinstance(fontes, list) else []
    fontes_validas = set()
    for ident in ids:
        if ident in chunks_por_id:
            chunk = chunks_por_id[ident]
            metadados = chunk.get('metadados', {})
            for origem in (chunk.get('fontes', []), metadados.get('fontes', [])):
                if isinstance(origem, str):
                    fontes_validas.add(origem)
                elif isinstance(origem, list):
                    for item in origem:
                        if isinstance(item, str):
                            fontes_validas.add(item)
                        elif isinstance(item, dict):
                            for chave in ('url', 'link'):
                                if isinstance(item.get(chave), str):
                                    fontes_validas.add(item[chave])
    citacoes_validas = (bool(ids) and bool(fontes)
                        and all(isinstance(x, str) and x in chunks_por_id for x in ids)
                        and all(isinstance(x, str) and x in fontes_validas for x in fontes))
    if obtido == 'recusa':
        citacoes_validas = not ids and not fontes
    if configuracao == 'qwen_puro':
        citacoes_validas = None
    relevantes = ids_esperados(g)
    cobertura = (len(set(ids) & set(relevantes)) / len(set(relevantes))) if relevantes and configuracao != 'qwen_puro' else None
    return {
        'status_esperado': esperado, 'status_obtido': obtido,
        'status_correto': status_correto,
        'estrutura_valida': validar_estrutura(resposta),
        'citacoes_validas': citacoes_validas,
        'cobertura_chunks_gabarito': cobertura,
        'recusa_correta': (obtido == 'recusa') if esperado == 'recusa' else None,
        'categoria_esperada': categoria_esperada(g),
    }


def proporcao(registros, campo):
    vals = [r['metricas'][campo] for r in registros if r['metricas'][campo] is not None]
    return round(100 * sum(vals) / len(vals), 2) if vals else None


def salvar_json(caminho, dados):
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    PASTA.mkdir(parents=True, exist_ok=True)
    casos = carregar_casos()
    print(f'{len(casos)} perguntas e gabaritos carregados.')
    chunks, embeddings = rag.carregar_base()
    chunks_por_id = {x['id']: x for x in chunks}
    modelo = SentenceTransformer(rag.MODELO_EMBEDDING)
    documentos, idf = rag.calcular_idf(chunks)
    cliente = ollama.Client(host=rag.OLLAMA_HOST)
    saidas = {nome: [] for nome in CONFIGURACOES}
    cache_busca = {}
    for i, caso in enumerate(casos, 1):
        pergunta = caso['pergunta']
        print(f'\n[{i}/{len(casos)}] {caso["id"]}: {pergunta[:85]}')
        try:
            cache_busca[caso['id']] = rag.buscar_contexto(
                pergunta, chunks, embeddings, modelo, documentos, idf)
        except Exception as e:
            raise RuntimeError(f'Falha na busca para a pergunta {caso["id"]}: {e}') from e
        for nome in CONFIGURACOES:
            inicio = time.perf_counter()
            erro = None
            try:
                if nome == 'qwen_puro':
                    resposta = qwen_puro(cliente, pergunta)
                elif nome == 'rag_simples':
                    resultados, semantica, palavras = cache_busca[caso['id']]
                    resposta = rag.gerar_resposta(pergunta, resultados, semantica, palavras, cliente)
                else:
                    resultados, semantica, palavras = cache_busca[caso['id']]
                    resposta = agente.gerar_resposta_agente(
                        pergunta, resultados, semantica, palavras, cliente, diagnostico=False)
            except Exception as e:
                erro = f'{type(e).__name__}: {e}'
                resposta = {'status': 'erro_execucao', 'resposta': '',
                            'fontes': [], 'chunks_utilizados': []}
            duracao = round(time.perf_counter() - inicio, 3)
            metricas = avaliar_resposta(resposta, caso['gabarito'], chunks_por_id, nome)
            if erro:
                metricas['erro'] = erro
            registro = {'id': caso['id'], 'pergunta': pergunta,
                        'gabarito': caso['gabarito'], 'resposta': resposta,
                        'metricas': metricas, 'duracao_segundos': duracao}
            saidas[nome].append(registro)
            salvar_json(PASTA / f'respostas_{nome}.json', saidas[nome])
            print(f'  {nome}: {resposta.get("status") if isinstance(resposta, dict) else "inválido"} ({duracao}s)')
    placar = {'data': datetime.now().isoformat(timespec='seconds'),
              'modelo': rag.OLLAMA_MODEL, 'total_perguntas': len(casos),
              'nota_metodologica': ('Status correto compara o status com o gabarito quando presente. '
                                    'Citações válidas verificam IDs e URLs, não a fidelidade semântica. '
                                    'Cobertura de chunks é calculada somente quando o gabarito informa IDs. '
                                    'Qualidade factual e segurança exigem revisão humana. Para Qwen puro, citações e cobertura são N/A. O gabarito mede adequação ao escopo documental do projeto, não conhecimento geral.'),
              'configuracoes': {}}
    for nome, registros in saidas.items():
        placar['configuracoes'][nome] = {
            'total': len(registros),
            'status_correto_pct': proporcao(registros, 'status_correto'),
            'estrutura_valida_pct': proporcao(registros, 'estrutura_valida'),
            'citacoes_validas_pct': proporcao(registros, 'citacoes_validas'),
            'recusas_corretas_pct': proporcao(registros, 'recusa_correta'),
            'cobertura_chunks_gabarito_pct': proporcao(registros, 'cobertura_chunks_gabarito'),
            'tempo_medio_segundos': round(sum(x['duracao_segundos'] for x in registros) / len(registros), 3)
            if registros else None,
            'erros_execucao': sum('erro' in x['metricas'] for x in registros),
        }
    salvar_json(PASTA / 'placar_configuracoes.json', placar)
    with (PASTA / 'placar_configuracoes.csv').open('w', encoding='utf-8-sig', newline='') as f:
        escritor = csv.writer(f, delimiter=';')
        escritor.writerow(['configuracao', 'total', 'status_correto_pct', 'estrutura_valida_pct',
                           'citacoes_validas_pct', 'recusas_corretas_pct',
                           'cobertura_chunks_gabarito_pct', 'tempo_medio_segundos', 'erros_execucao'])
        for nome, dados in placar['configuracoes'].items():
            escritor.writerow([nome] + [dados.get(c) for c in (
                'total', 'status_correto_pct', 'estrutura_valida_pct', 'citacoes_validas_pct',
                'recusas_corretas_pct', 'cobertura_chunks_gabarito_pct',
                'tempo_medio_segundos', 'erros_execucao')])
    print('\nPLACAR FINAL')
    for nome, dados in placar['configuracoes'].items():
        print(f'{nome}: status correto={dados["status_correto_pct"]}% | '
              f'estrutura={dados["estrutura_valida_pct"]}% | '
              f'citacoes={dados["citacoes_validas_pct"] if dados["citacoes_validas_pct"] is not None else "N/A"} | '
              f'recusas corretas={dados["recusas_corretas_pct"]}%')
    print(f'\nArquivos gerados em: {PASTA}')


if __name__ == '__main__':
    main()

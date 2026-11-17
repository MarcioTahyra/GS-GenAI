import json
import re
import ollama
from sentence_transformers import SentenceTransformer

try:
    from . import assistente_rag as rag
except ImportError:
    import assistente_rag as rag

CAMPOS = (
    'status', 'resposta', 'possivel_problema', 'servico_recomendado',
    'orientacoes_seguranca', 'pergunta_esclarecimento', 'fontes', 'chunks_utilizados'
)

REGRAS = {
    'bateria': {
        'sinais': ('bateria', 'painel', 'liga', 'partida', 'recarga'),
        'problema': 'Possível problema na bateria ou no sistema de recarga',
        'servico': 'Avaliação elétrica ou auxílio de bateria',
        'resposta': ('Os sinais relatados podem indicar um problema na bateria '
                     'ou no sistema de recarga, mas não confirmam um defeito. '
                     'Solicite assistência profissional para avaliar o veículo.'),
        'orientacoes': [
            'Evite procedimentos elétricos sem conhecimento adequado.',
            'Solicite assistência profissional se o veículo não ligar.'
        ],
    },
    'pneus': {
        'sinais': ('pneu', 'furou', 'furado', 'estepe', 'troca'),
        'problema': 'Possível pneu furado ou danificado',
        'servico': 'Auxílio para troca de pneu',
        'resposta': ('O relato é compatível com um pneu furado. '
                     'Priorize sua segurança e solicite auxílio para a troca '
                     'se não houver condições seguras de realizá-la.'),
        'orientacoes': [
            'Evite realizar a troca em local com risco de atropelamento.',
            'Solicite assistência se estiver em um local inseguro.'
        ],
    },
}

PADROES_BLOQUEADOS = (
    r'\bR\$\s*\d',
    r'\b(?:chega|chegará|atendimento em)\s+\d+\s*(?:minutos?|horas?)\b',
    r'\b(?:diagnóstico confirmado|com certeza é|garantidamente)\b',
)


def recusa(motivo):
    return rag.criar_recusa(motivo)


def auditar(resposta, selecionados):
    """Confere estrutura, IDs, URLs e afirmações proibidas; não verifica semântica livre."""
    if not isinstance(resposta, dict):
        return None, 'Resposta não é um objeto JSON'
    if resposta.get('status') not in ('respondido', 'esclarecer', 'recusa'):
        return None, 'Status inválido'
    if resposta['status'] == 'recusa':
        return recusa(str(resposta.get('resposta') or 'Sem evidências suficientes.')), None
    if not isinstance(resposta.get('resposta'), str) or not resposta['resposta'].strip():
        return None, 'Resposta vazia'
    orientacoes = resposta.get('orientacoes_seguranca')
    if not isinstance(orientacoes, list) or not all(isinstance(x, str) for x in orientacoes):
        return None, 'Orientações fora do formato esperado'
    ids = resposta.get('chunks_utilizados')
    if not isinstance(ids, list):
        return None, 'IDs de chunks fora do formato esperado'
    evidencias = {item['id']: item for item in selecionados}
    ids = list(dict.fromkeys(x for x in ids if isinstance(x, str) and x in evidencias))
    if not ids:
        return None, 'Sem chunks rastreáveis'
    fontes = []
    for identificador in ids:
        for url in evidencias[identificador].get('fontes', []):
            if isinstance(url, str) and url.startswith(('https://', 'http://')) and url not in fontes:
                fontes.append(url)
    if not fontes:
        return None, 'Sem URLs de fonte'
    texto = ' '.join([resposta['resposta'], str(resposta.get('possivel_problema') or ''),
                      str(resposta.get('servico_recomendado') or '')] + orientacoes)
    if any(re.search(p, texto, flags=re.IGNORECASE) for p in PADROES_BLOQUEADOS):
        return None, 'Preço, prazo ou diagnóstico definitivo não autorizado'
    saida = {campo: resposta.get(campo) for campo in CAMPOS}
    saida['chunks_utilizados'] = ids
    saida['fontes'] = fontes
    saida['orientacoes_seguranca'] = orientacoes
    for campo in ('possivel_problema', 'servico_recomendado', 'pergunta_esclarecimento'):
        if saida[campo] is not None and not isinstance(saida[campo], str):
            return None, f'Campo {campo} fora do formato esperado'
    return saida, None


def resposta_padronizada(categoria, selecionados):
    """Fallback conservador apenas para categorias e fatos explicitamente cobertos."""
    regra = REGRAS[categoria]
    evidencias = [x for x in selecionados if x.get('categoria') == categoria]
    if not evidencias:
        return None, 'Nenhum documento da categoria'
    texto = rag.normalizar_frase(' '.join(str(x.get('texto', '')) for x in evidencias))
    if not any(rag.normalizar_frase(termo) in texto for termo in regra['sinais']):
        return None, 'Evidências não sustentam a categoria'
    candidato = {
        'status': 'respondido',
        'resposta': regra['resposta'],
        'possivel_problema': regra['problema'],
        'servico_recomendado': regra['servico'],
        'orientacoes_seguranca': regra['orientacoes'],
        'pergunta_esclarecimento': None,
        'fontes': [],
        'chunks_utilizados': [x['id'] for x in evidencias],
    }
    return auditar(candidato, evidencias)




POLITICAS_SEGURANCA = {
    'temperatura': {
        'categoria': 'superaquecimento',
        'termos_evidencia': ('desligue o motor', 'não continue dirigindo', 'aguarde o sistema esfriar'),
        'resposta': 'Temperatura elevada pode indicar superaquecimento. Pare em local seguro, desligue o motor quando estiver em segurança, aguarde o sistema esfriar e solicite assistência profissional.',
        'problema': 'Possível superaquecimento do motor',
        'servico': 'Assistência mecânica ou guincho',
        'orientacoes': [
            'Procure parar em local seguro.',
            'Não continue dirigindo com o motor superaquecendo.',
            'Não abra o reservatório enquanto o sistema estiver quente.'
        ]
    },
    'reservatorio': {
        'categoria': 'superaquecimento',
        'termos_evidencia': ('nunca abra a tampa', 'causando queimaduras'),
        'resposta': 'Não é seguro abrir o reservatório de arrefecimento com o motor quente: o sistema pode estar pressurizado e liberar líquido ou vapor quente, causando queimaduras.',
        'problema': 'Risco de queimadura no sistema de arrefecimento quente',
        'servico': 'Avaliação do sistema de arrefecimento, se necessário',
        'orientacoes': [
            'Não abra a tampa do reservatório enquanto o sistema estiver quente.',
            'Aguarde o sistema esfriar e procure assistência profissional.'
        ]
    },
    'espera': {
        'categoria': 'seguranca',
        'termos_evidencia': ('local protegido', 'distante do fluxo de veículos'),
        'resposta': 'Depois de imobilizar o veículo, aguarde o atendimento em local protegido e distante do fluxo de veículos. Evite permanecer próximo à pista.',
        'problema': 'Veículo imobilizado em situação de emergência',
        'servico': 'Assistência veicular, conforme a situação',
        'orientacoes': [
            'Mantenha-se em local protegido, distante do fluxo de veículos.',
            'Evite permanecer próximo à pista.'
        ]
    }
}


def identificar_politica(consulta):
    t = rag.normalizar_frase(consulta)
    if ('reservatorio' in t or 'tampa' in t) and ('quente' in t or 'arrefecimento' in t):
        return 'reservatorio'
    if (('temperatura' in t and ('motor' in t or 'alta' in t))
            or 'motor superaquecendo' in t):
        return 'temperatura'
    if (('esperar' in t or 'aguardar' in t or 'onde devo ficar' in t)
            and ('ajuda' in t or 'emergencia' in t or 'veiculo' in t)):
        return 'espera'
    return None


def aplicar_politica_seguranca(consulta, resultados):
    nome = identificar_politica(consulta)
    if not nome:
        return None, None
    regra = POLITICAS_SEGURANCA[nome]

    chunks, _ = rag.carregar_base()
    evidencias = []
    for chunk in chunks:
        if chunk.get('metadados', {}).get('categoria') != regra['categoria']:
            continue
        texto = rag.normalizar_frase(chunk.get('texto', ''))
        if not any(rag.normalizar_frase(termo) in texto for termo in regra['termos_evidencia']):
            continue
        metadados = chunk.get('metadados', {})
        evidencias.append({
            'id': chunk['id'],
            'texto': chunk['texto'],
            'categoria': regra['categoria'],
            'documento': metadados.get('documento', ''),
            'fontes': metadados.get('fontes', []),
        })
    if not evidencias:
        return None, nome
    candidato = {
        'status': 'respondido',
        'resposta': regra['resposta'],
        'possivel_problema': regra['problema'],
        'servico_recomendado': regra['servico'],
        'orientacoes_seguranca': regra['orientacoes'],
        'pergunta_esclarecimento': None,
        'fontes': [],
        'chunks_utilizados': [x['id'] for x in evidencias],
    }
    validado, erro = auditar(candidato, evidencias)
    return validado, nome if not erro else f'{nome}: {erro}'


def gerar_resposta_agente(consulta, resultados, maior_similaridade,
                          maior_similaridade_palavras, cliente, diagnostico=False):

    politica, nome_politica = aplicar_politica_seguranca(consulta, resultados)
    if politica is not None:
        if diagnostico:
            print('Diagnóstico: política de segurança documentada aplicada:', nome_politica)
        return politica
    selecionados = rag.selecionar_contexto_compativel(consulta, resultados)
    if not selecionados:
        return recusa('O assunto não está documentado na base de conhecimento.')
    if (max(x['similaridade_semantica'] for x in selecionados) < rag.LIMIAR_SEMANTICO
            or maior_similaridade_palavras < rag.LIMIAR_PALAVRAS_CHAVE):
        return recusa('Não encontrei evidências suficientes para orientar com segurança.')

    categorias = rag.categorias_da_consulta(consulta) - {'seguranca'}
    categorias_com_evidencia = {x['categoria'] for x in selecionados} & categorias


    if len(categorias_com_evidencia) == 1:
        categoria = next(iter(categorias_com_evidencia))
        if categoria in REGRAS:
            resposta, erro = resposta_padronizada(categoria, selecionados)
            if resposta:
                if diagnostico:
                    print('Diagnóstico: política especializada aplicada:', categoria)
                return resposta
            if diagnostico:
                print('Diagnóstico: regra especializada não aplicável:', erro)

    candidato = rag.gerar_resposta(
        consulta, resultados, maior_similaridade, maior_similaridade_palavras, cliente
    )
    validado, erro = auditar(candidato, selecionados)
    if erro:
        if diagnostico:
            print('Diagnóstico: resposta RAG rejeitada:', erro)
        return recusa('Não foi possível validar uma orientação suficientemente fundamentada.')
    if diagnostico:
        print('Diagnóstico: RAG original com auditoria estrutural e de fontes.')
    return validado


def main():
    print('=' * 60)
    print('ASSISTENTE VEICULAR - RAG COM AGENTE ESPECIALIZADO V7')
    print('=' * 60)
    chunks, embeddings = rag.carregar_base()
    modelo = SentenceTransformer(rag.MODELO_EMBEDDING)
    documentos, idf = rag.calcular_idf(chunks)
    cliente = ollama.Client(host=rag.OLLAMA_HOST)
    print(f'{len(chunks)} chunks carregados | Modelo: {rag.OLLAMA_MODEL}')
    print("Digite 'sair' para encerrar.\n")
    while True:
        try:
            consulta = input('Descreva o problema do veículo: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if consulta.lower() in ('sair', 'exit', 'quit'):
            break
        if not consulta:
            continue
        try:
            resultados, semantica, palavras = rag.buscar_contexto(
                consulta, chunks, embeddings, modelo, documentos, idf
            )
            print('\nChunks recuperados:')
            for item in resultados:
                print(f"- {item['id']} (semântica: {item['similaridade_semantica']:.4f})")
            resposta = gerar_resposta_agente(
                consulta, resultados, semantica, palavras, cliente, diagnostico=True
            )
            print('\nRESPOSTA DO AGENTE')
            print(json.dumps(resposta, ensure_ascii=False, indent=4))
            print('\n' + '=' * 60 + '\n')
        except Exception as erro:
            print(f'Erro durante o processamento: {erro}\n')


if __name__ == '__main__':
    main()

from typing import TypedDict

from indexando_dados_qdrant_local import banco_qdrant

from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END


class MeuEstado(TypedDict):
    entrada_usuario: str
    documentos: list
    documentos_formatados: str
    resposta: str


def retriever_no(state: MeuEstado):
    db = banco_qdrant("meus_documentos")
    meu_retriever = db.as_retriever()
    documentos_retornados = meu_retriever.invoke(state["entrada_usuario"])
    return {"documentos": documentos_retornados}


def formata_documentos(state: MeuEstado):
    texto = "\n\n".join(doc.page_content for doc in state["documentos"])
    return {"documentos_formatados": texto}


def chatbot(state: MeuEstado):
    prompt_rag = f"""Você é um assistente útil que responde à perguntas do usuário com base no contexto fornecido.
Leia atentamente a dúvida do usuário e observe o contexto retornado de um banco vetorial de pesquisa utilizado como fonte\
de informação:

<contexto>
{state["documentos_formatados"]}
</contexto>

Caso você não saiba, responda que não tem informações sobre isso. Não invente informações."""

    model = ChatGroq(model="llama-3.3-70b-versatile")
    resposta_model = model.invoke([SystemMessage(content=prompt_rag), HumanMessage(content=state["entrada_usuario"])])

    return {"resposta": resposta_model.content}

## Inicializar o grafo
graph = StateGraph(MeuEstado)

## Adiciona os nós
graph.add_node("retriever_no", retriever_no)
graph.add_node("formata_documentos", formata_documentos)
graph.add_node("chatbot", chatbot)

## Adiciona as arestas
graph.add_edge(START, "retriever_no")
graph.add_edge("retriever_no", "formata_documentos")
graph.add_edge("formata_documentos", "chatbot")
graph.add_edge("chatbot", END)

## Compila o grafo
app = graph.compile()

## Invocando o grafo:
resp = app.invoke({"entrada_usuario": "Recebi uma cobrança indevida. O que devo fazer?"})

print(resp["resposta"])

# # Imprimindo a Imagem do Grafo:
# import io
# from PIL import Image
# img_bytes = app.get_graph(xray=1).draw_mermaid_png()
# img = Image.open(io.BytesIO(img_bytes))
# img.save('diagrama_workflow_game.png')
# img.show()

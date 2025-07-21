from dotenv import load_dotenv
load_dotenv()

from indexando_dados_qdrant_local import banco_qdrant


from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter


## Função auxiliar que formata o retorno do banco vetorial em uma string estruturada.
def cria_texto_dos_documentos_retornados(documentos):
    return "\n\n".join(doc.page_content for doc in documentos)


## Inicializando o Qdrant como banco vetorial:
db = banco_qdrant("meus_documentos")

meu_retriever = db.as_retriever()


prompt_rag  = """Você é um assistente útil que responde à perguntas do usuário com base no contexto fornecido.
Leia atentamente a dúvida do usuário e observe o contexto retornado de um banco vetorial de pesquisa utilizado como fonte\
de informação:

<contexto>
{documentos_formatados}
</contexto>

Caso você não saiba, responda que não tem informações sobre isso. Não invente informações.
"""

prompt_template = ChatPromptTemplate([("system", prompt_rag),
                                      ("human", "{entrada_usuario}")])


model = ChatGroq(model="llama-3.3-70b-versatile")


## Criando minha chain Retriever (precisa receber a entrada de usuário, então capturo ela do dict {"entrada_usuario": "..."}):
chain_retriever = itemgetter("entrada_usuario") | meu_retriever | cria_texto_dos_documentos_retornados


## Criando minha chain Principal unindo retriever e componentes de modelo, prompt e analisador de saida:
chain_rag = RunnableParallel({"entrada_usuario": itemgetter("entrada_usuario"),
                              "documentos_formatados": chain_retriever}) | prompt_template | model | StrOutputParser()

## Invocando a chain principal
resposta = chain_rag.invoke({"entrada_usuario": "Recebi uma cobrança indevida. O que devo fazer?"})


print(resposta)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory


# criar sistema de gestão do prompt: sistema + usuário + histórico + ia
prompt = ChatPromptTemplate.from_messages([
    ("system", "Atue como um assistente de IA útil"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{human_input}")])


# persistir todas as conversas baseadas em sessão do usuário em um banco de dados SQL
def get_session_history_db(session_id):
    return SQLChatMessageHistory(session_id, connection="sqlite:///memory.db")


# crie uma função de janela de buffer de memória para retornar as últimas K conversas
def memory_window(messages, k=10):
    return messages[-(k + 1):]


chatgpt = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key="SUA_CHAVE_AQUI")


llm_chain = (RunnablePassthrough.assign(history=lambda x: memory_window(x["history"]))
             | prompt
             | chatgpt
             | StrOutputParser())

conv_chain = RunnableWithMessageHistory(llm_chain, get_session_history_db, input_messages_key="human_input", history_messages_key="history")

print(conv_chain.invoke(input={"human_input": "Quais foram as duas perguntas que eu acabei de te fazer??"}, config={'configurable': { 'session_id': "ID_USUARIO"}}))
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Força o uso da API estável v1 para evitar erros 404 no v1beta
os.environ["GOOGLE_API_VERSION"] = "v1"

load_dotenv()

def test_connection():
    try:
        # Inicializa o modelo (Ajuste para o ID encontrado no passo anterior se necessário)
        model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

        print("--- Enviando requisição ao Gemini ---")
        result = model.invoke("Este é um teste. Se você recebeu a requisição responda 'Teste OK'.")

        #print(f" [V] Sucesso: {result.content}")
        print(result)

    except Exception as e:
        print(f"[X] Erro na configuração: {e}")


if __name__ == "__main__":
    test_connection()

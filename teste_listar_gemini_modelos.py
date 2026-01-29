import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def list_raw_models():
    try:
        # Inicializa o cliente oficial (SDK 2026)
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        print("--- Listando todos os modelos vinculados à sua chave ---")

        for model in client.models.list():
            print(f"ID: {model.name}")

    except Exception as e:
        print(f"Erro de conexão/autenticação: {e}")


if __name__ == "__main__":
    list_raw_models()

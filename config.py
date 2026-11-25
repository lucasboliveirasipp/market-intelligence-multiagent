"""Configurações centralizadas do Agente Inteligência de Mercado.

Inclui configuração de proxy corporativo, Azure OpenAI e Azure Cognitive Search.
"""

# Carrega variáveis do arquivo .env mais próximo (pasta do projeto)
def _carregar_env():
    from pathlib import Path
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for pasta in Path(__file__).resolve().parents:
        if (pasta / ".env").exists():
            load_dotenv(pasta / ".env")
            break


_carregar_env()


import os
import logging

from langchain_openai import AzureChatOpenAI

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAÇÃO DE PROXY CORPORATIVO
# ============================================================================

proxy_host = os.getenv("HTTP_PROXY_HOST", "")
proxy_port = os.getenv("HTTP_PROXY_PORT", "80")
if proxy_host:  # proxy opcional (defina HTTP_PROXY_HOST no .env)
    os.environ["HTTP_PROXY"] = f"http://{proxy_host}:{proxy_port}"
if proxy_host:  # proxy opcional (defina HTTP_PROXY_HOST no .env)
    os.environ["HTTPS_PROXY"] = f"http://{proxy_host}:{proxy_port}"
_no_proxy_default = "localhost,127.0.0.1,openai.azure.com,.openai.azure.com"
os.environ["NO_PROXY"] = os.getenv("NO_PROXY", _no_proxy_default)

# ============================================================================
# AZURE OPENAI - LLM Principal
# ============================================================================

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "",
)
AZURE_OPENAI_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
)
AZURE_OPENAI_DEPLOYMENT = os.getenv("GPT4", "gpt-4.1-mini")
AZURE_OPENAI_API_VERSION = os.getenv("API_VERSION", "2024-08-01-preview")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_KEY,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    openai_api_type="azure",
    temperature=0,
    max_retries=4,
    timeout=90,
)

# ============================================================================
# AZURE COGNITIVE SEARCH
# ============================================================================

AZURE_SEARCH_ENDPOINT = os.getenv(
    "AZURE_SEARCH_ENDPOINT",
    "",
)
AZURE_SEARCH_INDEX_NAME = os.getenv(
    "AZURE_SEARCH_INDEX_NAME",
    "inteligencia-mercado-index",
)
AZURE_SEARCH_API_KEY = os.getenv(
    "AZURE_SEARCH_API_KEY",
    "",
)
AZURE_SEARCH_SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG")
AZURE_SEARCH_DEFAULT_TOP_K = int(os.getenv("AZURE_SEARCH_TOP_K", "4"))

logger.info(f"Azure OpenAI: {AZURE_OPENAI_DEPLOYMENT}")
logger.info(f"Azure Search: {AZURE_SEARCH_INDEX_NAME}")

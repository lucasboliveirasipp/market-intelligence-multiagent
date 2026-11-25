"""
Agente Inteligência de Mercado - Interface Streamlit
Sistema de consulta inteligente sobre relatórios de mercado via Azure Search.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4
import logging
import re

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================
# LIMPEZA DE MÓDULOS (multi-agent environment)
# ============================================
_MODULES_TO_CLEAN = [
    "graph_builder",
    "agent_state",
    "node",
    "config",
    "prompts",
    "retriever",
]
for _mod_name in list(sys.modules.keys()):
    if any(_mod_name == m or _mod_name.startswith(m + ".") for m in _MODULES_TO_CLEAN):
        del sys.modules[_mod_name]

# ============================================
# CONFIGURAÇÃO DE PATH
# ============================================
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# ============================================
# IMPORTAÇÕES DO PROJETO
# ============================================
try:
    from .graph_builder import graph
    from .agent_state import AgentState
    logger.info("Módulos carregados (imports relativos)")
except ImportError:
    try:
        from graph_builder import graph
        from agent_state import AgentState
        logger.info("Módulos carregados (imports absolutos)")
    except Exception as e:
        import traceback
        logger.error(f"Erro ao importar módulos: {e}")
        st.error(f"Erro ao carregar módulos do projeto: {e}")
        st.code(traceback.format_exc(), language="text")
        st.stop()

# ============================================
# SISTEMA DE FEEDBACK (OPCIONAL)
# ============================================
FEEDBACK_AVAILABLE = False

try:
    from feedback_local import update_feedback
    from feedback_local import log_interaction
    FEEDBACK_AVAILABLE = True
except ImportError:
    def update_feedback(*args, **kwargs):
        pass

    def log_interaction(*args, **kwargs):
        pass

# ============================================
# CONSTANTES
# ============================================
AGENT_NAME = "Agente Inteligência de Mercado"


def _escape_dollar_for_markdown(text: str) -> str:
    """Escapa cifrões isolados para evitar renderização LaTeX no Streamlit."""
    if not text:
        return ""
    return re.sub(r"(?<!\$)\$(?!\$)", r"\$", text)


def _inject_styles() -> None:
    """Injeta CSS customizado com padrão Acme."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
        }

        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #2c3e50;
        }

        .main-header {
            background: linear-gradient(135deg, #1E3A8A 0%, #0F766E 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }

        .main-header h1 {
            color: white;
            margin-bottom: 0.5rem;
            font-size: 1.8rem;
            font-weight: 700;
        }

        .main-header .subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 1rem;
        }

        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            border-left: 4px solid #1E3A8A;
        }

        .info-card h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }

        .info-card p {
            color: #5a6c7d;
            line-height: 1.6;
            margin-bottom: 0.5rem;
        }

        .stButton > button {
            background: linear-gradient(135deg, #1E3A8A 0%, #0F766E 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(30, 58, 138, 0.4);
        }

        .sources-expander {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 0.5rem;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initialize_session_state() -> None:
    """Inicializa estados da sessão."""
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid4())
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def _render_sources(sources: list, query: str = "") -> None:
    """Renderiza fontes usadas na resposta."""
    if not sources:
        return

    with st.expander(f"Fontes utilizadas ({len(sources)})", expanded=False):
        for src in sources:
            src_id = src.get("id", "?")
            filepath = src.get("filepathField", "Documento")
            score = src.get("score")
            score_text = f" | Score: {score:.4f}" if score else ""
            st.markdown(f"**Fonte {src_id}**: {filepath}{score_text}")


def _render_sidebar() -> None:
    """Renderiza sidebar com informações do sistema."""
    with st.sidebar:
        st.markdown("## Sobre o Sistema")
        st.markdown(
            """
            <div class="info-card">
                <h3>Inteligência de Mercado</h3>
                <p><strong>Objetivo:</strong> Consulta inteligente sobre relatórios
                de mercado indexados no Azure Search.</p>
                <p><strong>Tecnologia:</strong> Azure OpenAI + Azure Cognitive Search</p>
                <p><strong>Arquitetura:</strong> Supervisor → Pesquisador → Redator → Editor</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("## Como usar")
        st.markdown(
            """
            <div class="info-card">
                <p>1. Digite sua pergunta no campo de chat abaixo.</p>
                <p>2. O agente pesquisará nos relatórios indexados.</p>
                <p>3. Um redator especializado formatará a resposta.</p>
                <p>4. As fontes serão citadas ao final de cada parágrafo.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("## Controles")
        if st.button("Nova Conversa", use_container_width=True):
            st.session_state["thread_id"] = str(uuid4())
            st.session_state["messages"] = []
            st.rerun()


def _run_agent(user_question: str) -> dict | None:
    """Executa o grafo LangGraph com a pergunta do usuário."""
    import time

    logger.info(f"Pergunta recebida: {user_question[:100]}...")

    try:
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_question)],
            "next": "",
        }

        config = {
            "configurable": {
                "thread_id": st.session_state["thread_id"],
            }
        }

        start_time = time.time()
        final_state = {}

        for state in graph.stream(initial_state, config, stream_mode="values"):
            final_state = state

        elapsed = time.time() - start_time
        logger.info(f"Pipeline concluído em {elapsed:.1f}s")

        return final_state

    except Exception as e:
        logger.error(f"Erro ao executar pipeline: {e}", exc_info=True)
        st.error(f"Erro ao processar sua pergunta: {e}")
        return None


def build_page(is_authenticated: bool) -> None:
    """
    Função principal - Constrói a página do agente de Inteligência de Mercado.
    Entry point para integração com o projeto maior.
    """
    _inject_styles()
    _initialize_session_state()

    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>Agente Inteligência de Mercado</h1>
            <div class="subtitle">
                Consulta inteligente sobre relatórios de mercado
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_authenticated:
        st.error("Você não está autenticado. Realize o login para acessar o sistema.")
        return

    # Sidebar
    _render_sidebar()

    # Exibir histórico de mensagens
    for i, msg in enumerate(st.session_state["messages"]):
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(_escape_dollar_for_markdown(msg["content"]))
            if role == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # Input do usuário
    if user_input := st.chat_input("Digite sua pergunta sobre inteligência de mercado..."):
        # Exibir mensagem do usuário
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Processar com o agente
        with st.chat_message("assistant"):
            with st.spinner("Pesquisando nos relatórios..."):
                final_state = _run_agent(user_input)

            if final_state:
                # Extrair resposta e fontes do último AIMessage com name="writer"
                messages = final_state.get("messages", [])
                response_text = ""
                sources = []

                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.name == "writer":
                        response_text = msg.content
                        sources = msg.additional_kwargs.get("sources", [])
                        break

                if not response_text:
                    # Fallback: pegar última AIMessage qualquer
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage):
                            response_text = msg.content
                            sources = msg.additional_kwargs.get("sources", [])
                            break

                if response_text:
                    st.markdown(_escape_dollar_for_markdown(response_text))
                    _render_sources(sources, query=user_input)

                    # Salvar no histórico
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources,
                    })

                    # Sistema de feedback
                    if FEEDBACK_AVAILABLE:
                        response_id = str(uuid4())
                        try:
                            log_interaction(
                                agent_name=AGENT_NAME,
                                thread_id=st.session_state["thread_id"],
                                response_id=response_id,
                                user_question=user_input,
                                chatbot_response=response_text,
                            )
                        except Exception:
                            pass
                else:
                    st.warning("Não foi possível gerar uma resposta. Tente reformular sua pergunta.")
            else:
                st.error("Erro ao processar a pergunta. Tente novamente.")


# Para execução standalone (desenvolvimento)
if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Inteligência de Mercado",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    build_page(is_authenticated=True)

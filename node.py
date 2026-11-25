"""Worker and supervisor nodes for the LangGraph hierarchy."""

from __future__ import annotations
from typing import Dict, List
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from .config import (
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_DEFAULT_TOP_K,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_SEARCH_SEMANTIC_CONFIG,
    llm,
)

from .prompts import (
    EDITOR_HUMAN_PROMPT,
    EDITOR_SYSTEM_PROMPT,
    RESEARCHER_HUMAN_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    SUPERVISOR_HUMAN_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    WRITER_HUMAN_PROMPT,
    WRITER_SYSTEM_PROMPT,
    get_contextual_researcher_prompt,
    get_contextual_writer_prompt,
)
 
from .retriever import AzureSearchParams, AzureSearchRetriever, _normalize_content

TEAM = ["researcher", "writer", "editor"]


def get_researcher_prompt(question: str) -> ChatPromptTemplate:
    """Retorna prompt do pesquisador baseado no contexto da pergunta."""
    system_prompt = get_contextual_researcher_prompt(question)
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", RESEARCHER_HUMAN_PROMPT),
    ])

def get_writer_prompt(question: str) -> ChatPromptTemplate:
    """Retorna prompt do escritor baseado no contexto da pergunta."""
    system_prompt = get_contextual_writer_prompt(question)
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", WRITER_HUMAN_PROMPT),
    ])

# Prompts estáticos para compatibilidade
researcher_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RESEARCHER_SYSTEM_PROMPT),
        ("human", RESEARCHER_HUMAN_PROMPT),
    ]
)

writer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", WRITER_HUMAN_PROMPT),
    ]
)

editor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", EDITOR_SYSTEM_PROMPT),
        ("human", EDITOR_HUMAN_PROMPT),
    ]
)

supervisor_prompt = (
    ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", SUPERVISOR_HUMAN_PROMPT),
        ]
    ).partial(options=", ".join(TEAM))
)


researcher_chain = researcher_prompt | llm
writer_chain = writer_prompt | llm
editor_chain = editor_prompt | llm


class SupervisorDecision(BaseModel):
    next: str


supervisor_chain = supervisor_prompt | llm.with_structured_output(SupervisorDecision)

retriever = AzureSearchRetriever(
    params=AzureSearchParams(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        api_key=AZURE_SEARCH_API_KEY,
        semantic_config=AZURE_SEARCH_SEMANTIC_CONFIG,
        top_k=AZURE_SEARCH_DEFAULT_TOP_K,
    )
)


def _latest_user_question(messages: List[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    raise ValueError("Nenhuma pergunta do usuario encontrada no historico.")


def _latest_writer_draft(messages: List[BaseMessage]) -> AIMessage:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.name == "writer_draft":
            return message
    raise ValueError("Nenhum rascunho do redator encontrado.")


def _preprocess_text(text: str) -> str:
    """
    Aplica correções automáticas básicas antes da edição manual.
    
    Args:
        text: Texto a ser pré-processado
        
    Returns:
        Texto com correções básicas aplicadas
    """
    import re
    
    if not text:
        return ""
    
    # Correções automáticas básicas
    corrections = [
        # Números ordinais
        (r'(\d+)\s+o\s+trimestre', r'\1º trimestre'),
        (r'(\d+)\s+o\s+semestre', r'\1º semestre'),
        
        # Moedas quebradas
        (r'R\\n(\d+)', r'R$ \1'),
        (r'R\$(\d+)', r'R$ \1'),
        (r'US\$(\d+)', r'US$ \1'),
        
        # Palavras coladas comuns
        (r'apresentandoum', 'apresentando um'),
        (r'Retornosobreo', 'Retorno sobre o'),
        (r'PatrimônioLíquido', 'Patrimônio Líquido'),
        (r'de29', 'de 29'),
        (r'de(\d+)', r'de \1'),
        
        # Siglas coladas
        (r'ROAEde(\d+)', r'ROAE de \1'),
        (r'EBITDAde', 'EBITDA de'),
        
        # Quebras de linha inadequadas
        (r'(\d+,\d+)\\n(bilhões|milhões)', r'\1 \2'),
        (r'R\\n(\d+)(milhões|bilhões)', r'R$ \1 \2'),
        
        # Citações de fontes
        (r'\(Fonte(\d+)\)', r'(Fonte \1)'),
        (r'\(Fonte(\d+)\s+e\s+Fonte(\d+)\)', r'(Fonte \1 e Fonte \2)'),
    ]
    
    processed_text = text
    for pattern, replacement in corrections:
        processed_text = re.sub(pattern, replacement, processed_text)
    
    return processed_text


def _polish_text(text: str) -> str:
    """
    Aplica correções de formatação ao texto usando o editor especializado.
    
    Args:
        text: Texto a ser corrigido
        
    Returns:
        Texto corrigido e formatado
    """
    if not text or not text.strip():
        return ""
    
    # Pré-processar com correções automáticas
    preprocessed = _preprocess_text(text)
    
    # Normalizar conteúdo preservando quebras de linha
    base = _normalize_content(preprocessed, preserve_newlines=True)
    if not base:
        return ""
    
    try:
        # Usar o editor chain com os prompts melhorados
        result = editor_chain.invoke({"draft": base})
        
        # Verificar se o resultado é válido
        if not result or not hasattr(result, 'content'):
            return base  # Retornar texto original se houver erro
            
        polished = result.content.strip()
        
        # Verificar se o texto foi realmente processado
        if not polished:
            return base  # Retornar texto original se estiver vazio
            
        return polished
        
    except Exception as e:
        # Log do erro (em produção, usar logging adequado)
        print(f"Erro na edição do texto: {e}")
        return base  # Retornar texto original em caso de erro


def researcher_node(state: Dict) -> Dict:
    question = _latest_user_question(state.get("messages", []))
    documents = retriever.invoke(question)

    context_blocks = []
    sources_payload: List[Dict] = []
    for index, doc in enumerate(documents, start=1):
        source_name = doc.metadata.get("filepathField") or f"Fonte {index}"
        context_blocks.append(
            f"Fonte {index}: {source_name}\nTrecho:\n{doc.page_content}"
        )
        sources_payload.append(
            {
                "id": index,
                "filepathField": source_name,
                "content": doc.page_content,
                "score": doc.metadata.get("score"),
                "contextual_score": doc.metadata.get("contextual_score"),
                "is_answer": doc.metadata.get("is_answer", False),
            }
        )

    context = "\n\n".join(context_blocks) if context_blocks else "Nenhum trecho encontrado."
    
    # Usar prompt contextual baseado na pergunta
    contextual_prompt = get_researcher_prompt(question)
    researcher_chain_contextual = contextual_prompt | llm
    
    response = researcher_chain_contextual.invoke({"question": question, "context": context})
    summary = _normalize_content(response.content, preserve_newlines=True)

    return {
        "messages": [
            AIMessage(
                content=summary,
                name="researcher",
                additional_kwargs={"sources": sources_payload},
            )
        ]
    }


def writer_node(state: Dict) -> Dict:
    messages: List[BaseMessage] = state.get("messages", [])
    question = _latest_user_question(messages)

    researcher_message = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, AIMessage) and message.name == "researcher"
        ),
        None,
    )

    notes = researcher_message.content if researcher_message else ""
    sources_payload = (
        researcher_message.additional_kwargs.get("sources", [])
        if researcher_message
        else []
    )

    # Usar prompt contextual baseado na pergunta
    contextual_prompt = get_writer_prompt(question)
    writer_chain_contextual = contextual_prompt | llm
    
    response = writer_chain_contextual.invoke({"question": question, "notes": notes})
    draft = _normalize_content(response.content, preserve_newlines=True)

    return {
        "messages": [
            AIMessage(
                content=draft,
                name="writer_draft",
                additional_kwargs={"sources": sources_payload},
            )
        ]
    }


def editor_node(state: Dict) -> Dict:
    messages: List[BaseMessage] = state.get("messages", [])
    draft_message = _latest_writer_draft(messages)

    polished = _polish_text(draft_message.content)

    return {
        "messages": [
            AIMessage(
                content=polished,
                name="writer",
                additional_kwargs=draft_message.additional_kwargs,
            )
        ]
    }


def supervisor_node(state: Dict) -> Dict:
    messages: List[BaseMessage] = state.get("messages", [])

    if messages and isinstance(messages[-1], AIMessage):
        last = messages[-1]
        if last.name == "writer":
            return {"next": "FINISH"}
        if last.name == "writer_draft":
            return {"next": "editor"}

    decision = supervisor_chain.invoke({"history": messages})
    candidate = (decision.next or "").strip().lower()

    if candidate in TEAM:
        next_agent = candidate
    elif candidate == "finish":
        next_agent = "FINISH"
    else:
        next_agent = "researcher"

    return {"next": next_agent}
 
"""Custom LangChain retriever backed by Azure Cognitive Search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import re
import unicodedata

import requests
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .prompts import (
    is_current_data_question,
    extract_specific_periods,
    extract_years_from_question
)

LETTER_CLASS = "A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF"
CONNECTOR_TERMS = [
    "que",
    "com",
    "para",
    "por",
    "nos",
    "nas",
    "no",
    "na",
    "dos",
    "das",
    "do",
    "da",
    "de",
    "em",
    "ao",
    "aos",
    "uma",
    "um",
    "\u00e9",
    "foi",
]
LETTER_RUN = re.compile(rf"[{LETTER_CLASS}]{6,}")
_ASSET_VOCAB: Set[str] = set()


def _strip_accents(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", value) if unicodedata.category(ch) != "Mn"
    )


def _load_asset_vocab() -> Set[str]:
    if _ASSET_VOCAB:
        return _ASSET_VOCAB

    root = Path(__file__).resolve().parent / "assets"
    if not root.exists():
        return _ASSET_VOCAB

    vocab: Set[str] = set()
    token_pattern = re.compile(rf"[{LETTER_CLASS}]{{2,}}")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in token_pattern.findall(text):
            vocab.add(_strip_accents(match.lower()))

    _ASSET_VOCAB.update(vocab)
    return _ASSET_VOCAB


def _segment_token(token: str) -> str:
    vocab = _load_asset_vocab()
    if not vocab:
        return token

    lowered = _strip_accents(token.lower())
    if lowered in vocab or len(lowered) < 10:
        return token

    length = len(lowered)
    dp: List[Optional[List[str]]] = [None] * (length + 1)
    dp[0] = []

    for i in range(length):
        pieces = dp[i]
        if pieces is None:
            continue
        for j in range(i + 2, length + 1):
            fragment = lowered[i:j]
            if fragment not in vocab:
                continue
            candidate = pieces + [token[i:j]]
            if dp[j] is None or len(candidate) < len(dp[j]):
                dp[j] = candidate

    parts = dp[-1]
    if not parts or len(parts) <= 1:
        return token
    return " ".join(parts)


def _segment_compound_words(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if any(ch.isdigit() for ch in token):
            return token
        return _segment_token(token)

    return LETTER_RUN.sub(repl, text)


def _normalize_content(text: str, *, preserve_newlines: bool = False) -> str:
    """Clean snippets returned by Azure Search for readability in Streamlit display."""

    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Fix specific problems that cause display issues in Streamlit
    # 1. Fix specific glued words that cause display problems (do this first)
    specific_fixes = [
        (r'Alémdisso', 'Além disso'),
        (r'aempresaapresentouumretornosobreopatrimôniolíquido', 'a empresa apresentou um retorno sobre o patrimônio líquido'),
        (r'\(ROAE\)de38', '(ROAE) de 38'),
        (r'R\s+139milhões', 'R$ 139 milhões'),
        (r'R\s+2,7\s+bilhões', 'R$ 2,7 bilhões'),
    ]
    
    for pattern, replacement in specific_fixes:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    # 2. Fix currency symbols (R$ problems) - comprehensive approach
    # Fix R without $ symbol
    normalized = re.sub(r'R(\d+)', r'R$ \1', normalized)  # R139 -> R$ 139
    normalized = re.sub(r'R\s+(\d+)', r'R$ \1', normalized)  # R 139 -> R$ 139
    
    # Fix R$ without space
    normalized = re.sub(r'R\$(\d+)', r'R$ \1', normalized)  # R$139 -> R$ 139
    normalized = re.sub(r'R\$(\d+,\d+)', r'R$ \1', normalized)  # R$139,50 -> R$ 139,50
    normalized = re.sub(r'R\$(\d+\.\d+)', r'R$ \1', normalized)  # R$139.50 -> R$ 139.50
    
    # Fix R$ with glued words (milhões, bilhões, etc.)
    normalized = re.sub(r'R\$\s*(\d+)(milhões)', r'R$ \1 \2', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'R\$\s*(\d+)(bilhões)', r'R$ \1 \2', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'R\$\s*(\d+)(milhão)', r'R$ \1 \2', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'R\$\s*(\d+)(bilhão)', r'R$ \1 \2', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'R\$\s*(\d+)(mil)', r'R$ \1 \2', normalized, flags=re.IGNORECASE)
    
    # Fix R$ with decimal numbers and glued words
    normalized = re.sub(r'R\$\s*(\d+,\d+)(milhões)', r'R$ \1 \2', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'R\$\s*(\d+\.\d+)(milhões)', r'R$ \1 \2', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'R\$\s*(\d+,\d+)(bilhões)', r'R$ \1 \2', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'R\$\s*(\d+\.\d+)(bilhões)', r'R$ \1 \2', normalized, flags=re.IGNORECASE)
    
    # Fix specific currency patterns from your examples
    normalized = re.sub(r'R(\d+),\s*(\d+)(milhões)', r'R$ \1,\2 \3', normalized)  # R889, 71milhões -> R$ 889,71 milhões
    normalized = re.sub(r'R(\d+\.\d+),\s*(\d+)(milhões)', r'R$ \1,\2 \3', normalized)  # R1.735, 16milhões -> R$ 1.735,16 milhões
    normalized = re.sub(r'R\s+(\d+\.\d+)\s+(\d+)\s+(mil)', r'R$ \1,\2 \3', normalized)  # R 889.710 mil -> R$ 889.710 mil
    
    # Fix R with decimal numbers and glued units (more specific patterns)
    normalized = re.sub(r'R(\d+\.\d+),\s*(\d+)(milhões)', r'R$ \1,\2 \3', normalized, flags=re.IGNORECASE)  # R1.735, 16milhões -> R$ 1.735,16 milhões
    normalized = re.sub(r'R(\d+\.\d+),\s*(\d+)(bilhões)', r'R$ \1,\2 \3', normalized, flags=re.IGNORECASE)  # R1.735, 16bilhões -> R$ 1.735,16 bilhões
    normalized = re.sub(r'R(\d+,\d+),\s*(\d+)(milhões)', r'R$ \1,\2 \3', normalized, flags=re.IGNORECASE)  # R1,735, 16milhões -> R$ 1,735,16 milhões
    
    # Fix R$ with decimal numbers and spaces before glued units
    normalized = re.sub(r'R\$\s*(\d+\.\d+),\s*(\d+)(milhões)', r'R$ \1,\2 \3', normalized, flags=re.IGNORECASE)  # R$1.735, 16milhões -> R$ 1.735,16 milhões
    normalized = re.sub(r'R\$\s+(\d+\.\d+),\s*(\d+)(milhões)', r'R$ \1,\2 \3', normalized, flags=re.IGNORECASE)  # R$ 1.735, 16milhões -> R$ 1.735,16 milhões
    normalized = re.sub(r'R\$\s*(\d+\.\d+),\s*(\d+)(bilhões)', r'R$ \1,\2 \3', normalized, flags=re.IGNORECASE)  # R$1.735, 16bilhões -> R$ 1.735,16 bilhões
    normalized = re.sub(r'R\$\s+(\d+\.\d+),\s*(\d+)(bilhões)', r'R$ \1,\2 \3', normalized, flags=re.IGNORECASE)  # R$ 1.735, 16bilhões -> R$ 1.735,16 bilhões
    
    # Fix more complex R$ patterns
    normalized = re.sub(r'R\$\s*(\d+)\s*,\s*(\d+)\s*(milhões)', r'R$ \1,\2 \3', normalized, flags=re.IGNORECASE)  # R$ 139, 50 milhões -> R$ 139,50 milhões
    normalized = re.sub(r'R\$\s*(\d+)\s*\.\s*(\d+)\s*(milhões)', r'R$ \1.\2 \3', normalized, flags=re.IGNORECASE)  # R$ 139. 50 milhões -> R$ 139.50 milhões
    normalized = re.sub(r'R\$\s*(\d+)\s*,\s*(\d+)\s*(bilhões)', r'R$ \1,\2 \3', normalized, flags=re.IGNORECASE)  # R$ 139, 50 bilhões -> R$ 139,50 bilhões
    normalized = re.sub(r'R\$\s*(\d+)\s*\.\s*(\d+)\s*(bilhões)', r'R$ \1.\2 \3', normalized, flags=re.IGNORECASE)  # R$ 139. 50 bilhões -> R$ 139.50 bilhões
    
    # Fix R$ with mixed separators and units
    normalized = re.sub(r'R\$\s*(\d+)\s*,\s*(\d+)\s*(mil)', r'R$ \1,\2 \3', normalized, flags=re.IGNORECASE)  # R$ 139, 50 mil -> R$ 139,50 mil
    normalized = re.sub(r'R\$\s*(\d+)\s*\.\s*(\d+)\s*(mil)', r'R$ \1.\2 \3', normalized, flags=re.IGNORECASE)  # R$ 139. 50 mil -> R$ 139.50 mil
    
    # Fix R$ with spaces in wrong places
    normalized = re.sub(r'R\$\s+(\d+)\s+\.\s+(\d+)', r'R$ \1.\2', normalized)  # R$ 1. 735 -> R$ 1.735
    normalized = re.sub(r'R\$\s+(\d+)\s+\.\s+(\d+)\s+(\d+)', r'R$ \1.\2\3', normalized)  # R$ 1. 735 16 -> R$ 1.73516
    normalized = re.sub(r'R\$\s+(\d+)\s+,\s+(\d+)', r'R$ \1,\2', normalized)  # R$ 1, 50 -> R$ 1,50
    
    # Fix R$ with spaces in decimal separators (more comprehensive)
    normalized = re.sub(r'R\$\s+(\d+)\s*\.\s+(\d+)', r'R$ \1.\2', normalized)  # R$ 1. 735 -> R$ 1.735
    normalized = re.sub(r'R\$\s+(\d+)\s*,\s+(\d+)', r'R$ \1,\2', normalized)  # R$ 1, 50 -> R$ 1,50
    normalized = re.sub(r'R\$\s+(\d+)\s*\.\s+(\d+)\s+(\d+)', r'R$ \1.\2\3', normalized)  # R$ 1. 735 16 -> R$ 1.73516
    
    # Fix R$ with multiple spaces
    normalized = re.sub(r'R\$\s+(\d+)', r'R$ \1', normalized)  # R$  139 -> R$ 139
    normalized = re.sub(r'R\$\s+(\d+,\d+)', r'R$ \1', normalized)  # R$  139,50 -> R$ 139,50
    normalized = re.sub(r'R\$\s+(\d+\.\d+)', r'R$ \1', normalized)  # R$  139.50 -> R$ 139.50
    
    # Final cleanup for R$ formatting issues
    # Fix R$ followed by any whitespace and numbers
    normalized = re.sub(r'R\$\s*(\d+(?:[.,]\d+)?)', r'R$ \1', normalized)  # R$139 -> R$ 139, R$139,50 -> R$ 139,50
    # Fix R$ with extra spaces before units
    normalized = re.sub(r'R\$\s+(\d+(?:[.,]\d+)?)\s+(milhões|bilhões|milhão|bilhão|mil)', r'R$ \1 \2', normalized, flags=re.IGNORECASE)
    # Fix R$ with glued units (no space)
    normalized = re.sub(r'R\$\s*(\d+(?:[.,]\d+)?)(milhões|bilhões|milhão|bilhão|mil)', r'R$ \1 \2', normalized, flags=re.IGNORECASE)
    
    # Fix specific problematic patterns
    # Fix R$ with decimal numbers and spaces in wrong places
    normalized = re.sub(r'R\$\s+(\d+\.\d+)\s+(\d+)\s+(milhões|bilhões|milhão|bilhão|mil)', r'R$ \1\2 \3', normalized, flags=re.IGNORECASE)  # R$ 1.735 16 milhões -> R$ 1.73516 milhões
    normalized = re.sub(r'R\$\s+(\d+\.\d+)\s+(\d+)(milhões|bilhões|milhão|bilhão|mil)', r'R$ \1\2 \3', normalized, flags=re.IGNORECASE)  # R$ 1.735 16milhões -> R$ 1.73516 milhões
    # Fix R$ with comma and spaces in wrong places
    normalized = re.sub(r'R\$\s+(\d+,\d+)\s+(\d+)\s+(milhões|bilhões|milhão|bilhão|mil)', r'R$ \1\2 \3', normalized, flags=re.IGNORECASE)  # R$ 1,735 16 milhões -> R$ 1,73516 milhões
    normalized = re.sub(r'R\$\s+(\d+,\d+)\s+(\d+)(milhões|bilhões|milhão|bilhão|mil)', r'R$ \1\2 \3', normalized, flags=re.IGNORECASE)  # R$ 1,735 16milhões -> R$ 1,73516 milhões
    
    # Fix R$ with spaces between decimal parts (more aggressive)
    normalized = re.sub(r'R\$\s+(\d+)\s*\.\s*(\d+)\s+(\d+)(milhões|bilhões|milhão|bilhão|mil)', r'R$ \1.\2\3 \4', normalized, flags=re.IGNORECASE)  # R$ 1. 735 16 milhões -> R$ 1.73516 milhões
    normalized = re.sub(r'R\$\s+(\d+)\s*,\s*(\d+)\s+(\d+)(milhões|bilhões|milhão|bilhão|mil)', r'R$ \1,\2\3 \4', normalized, flags=re.IGNORECASE)  # R$ 1, 735 16 milhões -> R$ 1,73516 milhões
    # Fix R$ with spaces between decimal parts without units
    normalized = re.sub(r'R\$\s+(\d+)\s*\.\s*(\d+)\s+(\d+)', r'R$ \1.\2\3', normalized)  # R$ 1. 735 16 -> R$ 1.73516
    normalized = re.sub(r'R\$\s+(\d+)\s*,\s*(\d+)\s+(\d+)', r'R$ \1,\2\3', normalized)  # R$ 1, 735 16 -> R$ 1,73516

    # 3. Fix fonte references
    normalized = re.sub(r"(?i)(Fonte)(\d+)", r"\1 \2", normalized)

    # 4. Ensure spacing around punctuation and numeric markers common in citations.
    normalized = re.sub(r"([,.;:!?])(?!\s)", r"\1 ", normalized)
    normalized = re.sub(r"(?<=\w)\(", " (", normalized)
    normalized = re.sub(r"\)(?=\w)", ") ", normalized)

    # 5. Fix common glued words with millions/billions (be more specific)
    normalized = re.sub(r'([a-záàâãéêíóôõúç])(milhões)', r'\1 \2', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'([a-záàâãéêíóôõúç])(bilhões)', r'\1 \2', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'([a-záàâãéêíóôõúç])(milhão)', r'\1 \2', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'([a-záàâãéêíóôõúç])(bilhão)', r'\1 \2', normalized, flags=re.IGNORECASE)

    # 6. Separate glued digits and currency markers from words (be more conservative)
    # Only separate when there's no space and it's not a decimal number
    normalized = re.sub(r'(?<=\d)(?=[a-zA-Záàâãéêíóôõúç])', ' ', normalized)  # 90k -> 90 k
    normalized = re.sub(r'(?<=[a-zA-Záàâãéêíóôõúç])(?=\d)', ' ', normalized)  # k90 -> k 90
    normalized = re.sub(r"(?i)(?<=\w)R(?=\s*\d)", " R", normalized)
    normalized = re.sub(r"(?<=R)(?=\d)", " ", normalized)
    
    # 7. Fix decimal numbers that got incorrectly separated
    normalized = re.sub(r'(\d+),\s+(\d+)', r'\1,\2', normalized)  # 53, 9 -> 53,9

    # 7. Insert spaces before common connectors accidentally glued to preceding words.
    for term in sorted(CONNECTOR_TERMS, key=len, reverse=True):
        pattern = re.compile(
            rf"([{LETTER_CLASS}])({term})(?=[{LETTER_CLASS}])",
            re.IGNORECASE,
        )
        normalized = pattern.sub(lambda m: f"{m.group(1)} {m.group(2)}", normalized)

    # 8. Segment compound words (conservative approach)
    normalized = _segment_compound_words(normalized)

    if preserve_newlines:
        lines: List[str] = []
        for line in normalized.split("\n"):
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    return re.sub(r"\s+", " ", normalized).strip()


@dataclass
class AzureSearchParams:
    endpoint: str
    index_name: str
    api_key: str
    semantic_config: Optional[str] = None
    top_k: int = 4


class AzureSearchRetriever(BaseRetriever):
    """Simple retriever that queries Azure Cognitive Search via REST API."""

    params: AzureSearchParams

    def _extract_periods_from_content(self, content: str) -> List[str]:
        """Extrai períodos do conteúdo (ex: 1T24, 2T25, etc.)."""
        period_patterns = [
            r'(\d{1,2})[Tt](\d{2})',  # 1T24, 2T25, etc.
            r'(\d{1,2})º?\s*trimestre\s*de\s*(\d{4})',  # 1º trimestre de 2024
            r'(\d{1,2})º?\s*trimestre\s*(\d{4})',  # 1º trimestre 2024
        ]
        
        periods = []
        for pattern in period_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    quarter, year = match
                    periods.append(f"{quarter}T{year}")
                else:
                    periods.append(match)
        
        return periods

    def _extract_years_from_content(self, content: str) -> List[int]:
        """Extrai anos do conteúdo."""
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, content)
        
        # Também extrair anos de períodos como 1T25, 2T24, etc.
        period_pattern = r'(\d{1,2})[Tt](\d{2})'
        period_matches = re.findall(period_pattern, content)
        for quarter, year in period_matches:
            full_year = int(f"20{year}")
            if 2020 <= full_year <= 2030:
                years.append(str(full_year))
        
        return [int(year) for year in years if 2020 <= int(year) <= 2030]

    def _get_priority_periods_for_query(self, query: str) -> List[str]:
        """Retorna períodos prioritários baseados na pergunta."""
        if is_current_data_question(query):
            # Para perguntas atuais, priorizar 2025 primeiro, depois 2024
            return ["4T25", "3T25", "2T25", "1T25", "4T24", "3T24", "2T24", "1T24"]
        
        specific_periods = extract_specific_periods(query)
        if specific_periods:
            return specific_periods
        
        years = extract_years_from_question(query)
        if years:
            # Se menciona ano específico, priorizar esse ano
            year = max(years)
            return [f"4T{year-2000}", f"3T{year-2000}", f"2T{year-2000}", f"1T{year-2000}"]
        
        return []

    def _enhance_query_for_context(self, query: str) -> str:
        """Melhora a query baseada no contexto da pergunta."""
        priority_periods = self._get_priority_periods_for_query(query)
        
        if priority_periods:
            # Adicionar períodos prioritários à query
            periods_str = " ".join(priority_periods)
            enhanced_query = f"{query} {periods_str}"
            return enhanced_query
        
        return query

    def _calculate_contextual_score(self, content: str, filepath: str, question: str) -> float:
        """Calcula score baseado no contexto da pergunta."""
        score = 0.0
        
        # Extrair períodos e anos do conteúdo
        content_periods = self._extract_periods_from_content(content)
        content_years = self._extract_years_from_content(content)
        
        # Extrair períodos e anos da pergunta
        question_periods = extract_specific_periods(question)
        question_years = extract_years_from_question(question)
        
        # Score por períodos específicos mencionados na pergunta
        if question_periods:
            for q_period in question_periods:
                if q_period in content_periods:
                    score += 100.0  # Match exato de período
                elif any(q_period in cp for cp in content_periods):
                    score += 50.0   # Match parcial de período
        
        # Score por anos específicos mencionados na pergunta
        if question_years and content_years:
            for q_year in question_years:
                if q_year in content_years:
                    score += 80.0   # Match exato de ano
                elif abs(q_year - max(content_years)) <= 1:
                    score += 40.0   # Ano próximo
        
        # Score por recência (apenas se pergunta for sobre dados atuais)
        if is_current_data_question(question):
            # Priorizar por trimestre mais recente
            if 2025 in content_years:
                # Dentro de 2025, priorizar trimestres mais recentes
                if any(period in content_periods for period in ['4T25', '3T25', '2T25', '1T25']):
                    if '4T25' in content_periods:
                        score += 120.0  # 4T25 é o mais recente
                    elif '3T25' in content_periods:
                        score += 110.0
                    elif '2T25' in content_periods:
                        score += 100.0
                    elif '1T25' in content_periods:
                        score += 90.0
                else:
                    score += 80.0  # 2025 sem trimestre específico
            elif 2024 in content_years:
                # Dentro de 2024, priorizar trimestres mais recentes
                if any(period in content_periods for period in ['4T24', '3T24', '2T24', '1T24']):
                    if '4T24' in content_periods:
                        score += 70.0  # 4T24 é o mais recente de 2024
                    elif '3T24' in content_periods:
                        score += 60.0
                    elif '2T24' in content_periods:
                        score += 50.0
                    elif '1T24' in content_periods:
                        score += 40.0
                else:
                    score += 30.0  # 2024 sem trimestre específico
            elif 2023 in content_years:
                score += 20.0
        
        # Score por nome do arquivo (assumindo que contém período)
        for q_period in question_periods:
            if q_period in filepath:
                score += 60.0
        
        for q_year in question_years:
            if str(q_year) in filepath:
                score += 40.0
        
        # Score por palavras-chave de recência (apenas se pergunta for sobre dados atuais)
        if is_current_data_question(question):
            recency_keywords = [
                'atual', 'recente', 'último', 'mais recente', 'atualização',
                'novo', 'lançamento', 'lançou', 'apresentou'
            ]
            
            content_lower = content.lower()
            for keyword in recency_keywords:
                if keyword in content_lower:
                    score += 5.0
        
        return score

    def _do_search(self, query: str, top_k: int = None) -> dict:
        """Executa busca no Azure Search com fallback automático sem semantic search."""
        url = (
            f"{self.params.endpoint}/indexes/{self.params.index_name}/docs/search"
            "?api-version=2023-11-01"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": self.params.api_key,
        }

        payload: dict = {
            "search": query,
            "top": top_k or self.params.top_k,
            "select": "content,filepathField",
        }

        if self.params.semantic_config:
            payload.update({
                "queryType": "semantic",
                "semanticConfiguration": self.params.semantic_config,
                "queryLanguage": "pt-BR",
                "captions": "extractive|highlight-true",
                "answers": "extractive|count-3",
            })

        response = requests.post(url, headers=headers, json=payload, timeout=15)

        # Se falhar com semantic params, retry sem eles
        if response.status_code != 200 and self.params.semantic_config:
            payload.pop("queryType", None)
            payload.pop("semanticConfiguration", None)
            payload.pop("queryLanguage", None)
            payload.pop("captions", None)
            payload.pop("answers", None)
            response = requests.post(url, headers=headers, json=payload, timeout=15)

        response.raise_for_status()
        return response.json()

    def _search_with_priority_periods(self, query: str, priority_periods: List[str]) -> List[Document]:
        """Busca documentos priorizando períodos específicos."""
        all_documents = []

        for period in priority_periods[:3]:
            period_query = f"{query} {period}"

            try:
                body = self._do_search(period_query)

                for item in body.get("value", []):
                    content = _normalize_content(item.get("content") or "")
                    filepath = item.get("filepathField", "")

                    contextual_score = self._calculate_contextual_score(content, filepath, query)

                    if period in content or period in filepath:
                        contextual_score += 50.0

                    metadata = {
                        "filepathField": filepath,
                        "score": item.get("@search.score"),
                        "contextual_score": contextual_score,
                        "search_period": period,
                    }

                    captions = item.get("@search.captions") or []
                    if not content and captions:
                        content = _normalize_content(captions[0].get("text", ""))

                    all_documents.append(Document(page_content=content, metadata=metadata))

            except Exception as e:
                print(f"Erro na busca para período {period}: {e}")
                continue

        return all_documents

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # Obter períodos prioritários para a query
        priority_periods = self._get_priority_periods_for_query(query)
        
        # Se há períodos prioritários, usar busca inteligente
        if priority_periods and is_current_data_question(query):
            documents = self._search_with_priority_periods(query, priority_periods)
            
            # Remover duplicatas baseado no filepath
            seen_files = set()
            unique_documents = []
            for doc in documents:
                filepath = doc.metadata.get("filepathField", "")
                if filepath not in seen_files:
                    seen_files.add(filepath)
                    unique_documents.append(doc)
            
            # Ordenar por score contextual e limitar resultados
            unique_documents.sort(key=lambda doc: doc.metadata.get("contextual_score", 0), reverse=True)
            return unique_documents[:self.params.top_k]
        
        # Busca tradicional para outros casos
        enhanced_query = self._enhance_query_for_context(query)

        try:
            body = self._do_search(enhanced_query)
        except requests.HTTPError as exc:
            resp_body = exc.response.text if exc.response is not None else ""
            raise RuntimeError(
                "Azure Cognitive Search request failed "
                f"(status={exc.response.status_code if exc.response else 'unknown'}): {resp_body}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Azure Cognitive Search request failed: {exc}") from exc
        documents: List[Document] = []

        for item in body.get("value", []):
            content = _normalize_content(item.get("content") or "")
            filepath = item.get("filepathField", "")
            
            # Calcular score contextual
            contextual_score = self._calculate_contextual_score(content, filepath, query)
            
            metadata = {
                "filepathField": filepath,
                "score": item.get("@search.score"),
                "contextual_score": contextual_score,
            }

            captions = item.get("@search.captions") or []
            if not content and captions:
                content = _normalize_content(captions[0].get("text", ""))

            documents.append(
                Document(page_content=content, metadata=metadata),
            )
        
        # Ordenar documentos por score contextual (maior primeiro)
        documents.sort(key=lambda doc: doc.metadata.get("contextual_score", 0), reverse=True)

        for answer in body.get("@search.answers", []):
            text = _normalize_content(answer.get("text", ""))
            if not text:
                continue
            metadata = {
                "filepathField": answer.get("key", "Resumo do mecanismo de busca"),
                "score": answer.get("score"),
                "is_answer": True,
            }
            documents.append(Document(page_content=text, metadata=metadata))

        return documents
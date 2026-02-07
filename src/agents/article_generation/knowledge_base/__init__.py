"""Knowledge-base integration package."""

from src.agents.article_generation.knowledge_base.indexer import KnowledgeBaseIndexer
from src.agents.article_generation.knowledge_base.retriever import HaystackKnowledgeBaseRetriever

__all__ = ["KnowledgeBaseIndexer", "HaystackKnowledgeBaseRetriever"]

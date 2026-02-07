"""Factory utilities for building the article-generation orchestrator."""

from pathlib import Path

from src.agents.article_generation.article_review.agent import ArticleReviewAgent
from src.agents.article_generation.chief_editor.bullet_parser import ArticleReviewBulletParser
from src.agents.article_generation.chief_editor.institutional_memory import InstitutionalMemoryStore
from src.agents.article_generation.chief_editor.orchestrator import ChiefEditorOrchestrator
from src.agents.article_generation.chief_editor.output_handler import OutputHandler
from src.agents.article_generation.concern_mapping.agent import ConcernMappingAgent
from src.agents.article_generation.knowledge_base.indexer import KnowledgeBaseIndexer
from src.agents.article_generation.knowledge_base.retriever import HaystackKnowledgeBaseRetriever
from src.agents.article_generation.llm_client import LiteLLMClient
from src.agents.article_generation.perplexity_client import PerplexityHTTPClient
from src.agents.article_generation.prompts.loader import PromptLoader
from src.agents.article_generation.specialists.attribution.agent import AttributionAgent
from src.agents.article_generation.specialists.evidence_finding.agent import EvidenceFindingAgent
from src.agents.article_generation.specialists.fact_check.agent import FactCheckAgent
from src.agents.article_generation.specialists.opinion.agent import OpinionAgent
from src.agents.article_generation.specialists.style_review.agent import StyleReviewAgent
from src.agents.article_generation.writer.agent import WriterAgent
from src.config import Config


def build_chief_editor_orchestrator(*, config: Config) -> ChiefEditorOrchestrator:
    """Build a fully wired chief editor orchestrator from configuration."""
    article_generation_config = config.get_article_generation_config()

    prompt_root_dir = Path(article_generation_config.editor.prompts.root_dir)
    prompt_loader = PromptLoader(root_dir=prompt_root_dir)
    llm_client = LiteLLMClient()

    institutional_memory = InstitutionalMemoryStore(
        data_dir=Path(article_generation_config.institutional_memory.data_dir),
        fact_checking_subdir=article_generation_config.institutional_memory.fact_checking_subdir,
        evidence_finding_subdir=article_generation_config.institutional_memory.evidence_finding_subdir,
    )

    output_handler = OutputHandler(
        final_articles_dir=Path(article_generation_config.editor.output.final_articles_dir),
        run_artifacts_dir=Path(article_generation_config.editor.output.run_artifacts_dir),
        save_intermediate_results=article_generation_config.editor.output.save_intermediate_results,
    )

    writer_agent = WriterAgent(
        llm_config=article_generation_config.agents.writer_llm,
        config=config,
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        writer_prompt_file=article_generation_config.editor.prompts.writer_prompt_file,
        revision_prompt_file=article_generation_config.editor.prompts.revision_prompt_file,
    )

    article_review_agent = ArticleReviewAgent(
        llm_config=article_generation_config.agents.article_review_llm,
        config=config,
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        prompt_file=article_generation_config.editor.prompts.article_review_prompt_file,
    )

    concern_mapping_agent = ConcernMappingAgent(
        llm_config=article_generation_config.agents.concern_mapping_llm,
        config=config,
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        prompt_file=article_generation_config.editor.prompts.concern_mapping_prompt_file,
    )

    kb_config = article_generation_config.knowledge_base
    kb_indexer = KnowledgeBaseIndexer(
        data_dir=Path(kb_config.data_dir),
        index_dir=Path(kb_config.index_dir),
        chunk_size_tokens=kb_config.chunk_size_tokens,
        chunk_overlap_tokens=kb_config.chunk_overlap_tokens,
        embedding_provider=kb_config.embedding.provider,
        embedding_model_name=kb_config.embedding.model_name,
        embedding_api_base=kb_config.embedding.api_base,
        embedding_api_key=kb_config.embedding.api_key,
        embedding_timeout_seconds=kb_config.embedding.timeout_seconds,
        encoding_name=config.getEncodingName(),
    )
    kb_index_version = kb_indexer.ensure_index()
    kb_retriever = HaystackKnowledgeBaseRetriever(
        index_dir=Path(kb_config.index_dir),
        embedding_provider=kb_config.embedding.provider,
        embedding_model_name=kb_config.embedding.model_name,
        embedding_api_base=kb_config.embedding.api_base,
        embedding_api_key=kb_config.embedding.api_key,
        embedding_timeout_seconds=kb_config.embedding.timeout_seconds,
    )

    perplexity_client = PerplexityHTTPClient(
        api_base=article_generation_config.perplexity.api_base,
        api_key=article_generation_config.perplexity.api_key,
    )

    fact_check_agent = FactCheckAgent(
        llm_config=article_generation_config.agents.specialists.fact_check_llm,
        config=config,
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        specialists_dir=article_generation_config.editor.prompts.specialists_dir,
        prompt_file="fact_check.md",
        knowledge_base_retriever=kb_retriever,
        institutional_memory=institutional_memory,
        kb_index_version=kb_index_version,
        kb_timeout_seconds=kb_config.timeout_seconds,
    )

    evidence_finding_agent = EvidenceFindingAgent(
        llm_config=article_generation_config.agents.specialists.evidence_finding_llm,
        config=config,
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        specialists_dir=article_generation_config.editor.prompts.specialists_dir,
        prompt_file="evidence_finding.md",
        perplexity_client=perplexity_client,
        perplexity_model=article_generation_config.perplexity.model,
        institutional_memory=institutional_memory,
    )

    opinion_agent = OpinionAgent(
        llm_config=article_generation_config.agents.specialists.opinion_llm,
        config=config,
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        specialists_dir=article_generation_config.editor.prompts.specialists_dir,
        prompt_file="opinion.md",
    )

    attribution_agent = AttributionAgent(
        llm_config=article_generation_config.agents.specialists.attribution_llm,
        config=config,
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        specialists_dir=article_generation_config.editor.prompts.specialists_dir,
        prompt_file="attribution.md",
    )

    style_review_agent = StyleReviewAgent(
        llm_config=article_generation_config.agents.specialists.style_review_llm,
        config=config,
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        specialists_dir=article_generation_config.editor.prompts.specialists_dir,
        prompt_file="style_review.md",
    )

    return ChiefEditorOrchestrator(
        config=config,
        writer_agent=writer_agent,
        article_review_agent=article_review_agent,
        concern_mapping_agent=concern_mapping_agent,
        fact_check_agent=fact_check_agent,
        evidence_finding_agent=evidence_finding_agent,
        opinion_agent=opinion_agent,
        attribution_agent=attribution_agent,
        style_review_agent=style_review_agent,
        bullet_parser=ArticleReviewBulletParser(),
        institutional_memory=institutional_memory,
        output_handler=output_handler,
    )

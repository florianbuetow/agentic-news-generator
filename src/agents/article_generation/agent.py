"""Factory utilities for building the article-generation orchestrator."""

from src.agents.article_generation.article_review.agent import ArticleReviewAgent
from src.agents.article_generation.article_review.mock_agent import MockArticleReviewAgent
from src.agents.article_generation.base import (
    ArticleReviewAgentProtocol,
    ConcernMappingAgentProtocol,
    SpecialistAgentProtocol,
    WriterAgentProtocol,
)
from src.agents.article_generation.chief_editor.bullet_parser import ArticleReviewBulletParser
from src.agents.article_generation.chief_editor.institutional_memory import InstitutionalMemoryStore
from src.agents.article_generation.chief_editor.orchestrator import ChiefEditorOrchestrator
from src.agents.article_generation.chief_editor.output_handler import OutputHandler
from src.agents.article_generation.concern_mapping.agent import ConcernMappingAgent
from src.agents.article_generation.concern_mapping.mock_agent import MockConcernMappingAgent
from src.agents.article_generation.knowledge_base.indexer import KnowledgeBaseIndexer
from src.agents.article_generation.knowledge_base.retriever import HaystackKnowledgeBaseRetriever
from src.agents.article_generation.llm_client import LiteLLMClient
from src.agents.article_generation.perplexity_client import PerplexityHTTPClient
from src.agents.article_generation.prompts.loader import PromptLoader
from src.agents.article_generation.specialists.attribution.agent import AttributionAgent
from src.agents.article_generation.specialists.attribution.mock_agent import MockAttributionAgent
from src.agents.article_generation.specialists.evidence_finding.agent import EvidenceFindingAgent
from src.agents.article_generation.specialists.evidence_finding.mock_agent import MockEvidenceFindingAgent
from src.agents.article_generation.specialists.fact_check.agent import FactCheckAgent
from src.agents.article_generation.specialists.fact_check.mock_agent import MockFactCheckAgent
from src.agents.article_generation.specialists.opinion.agent import OpinionAgent
from src.agents.article_generation.specialists.opinion.mock_agent import MockOpinionAgent
from src.agents.article_generation.specialists.style_review.agent import StyleReviewAgent
from src.agents.article_generation.specialists.style_review.mock_agent import MockStyleReviewAgent
from src.agents.article_generation.writer.agent import WriterAgent
from src.agents.article_generation.writer.mock_agent import MockWriterAgent
from src.config import Config


def _check_agent_name(agent_name: str, label: str) -> None:
    """Validate that agent_name is 'mock' or 'default'."""
    if agent_name not in ("mock", "default"):
        raise ValueError(f"Unknown {label} agent_name: '{agent_name}'")


def build_chief_editor_orchestrator(*, config: Config) -> ChiefEditorOrchestrator:
    """Build a fully wired chief editor orchestrator from configuration."""
    article_generation_config = config.get_article_generation_config()

    prompts_config = article_generation_config.editor.prompts
    prompt_loader = PromptLoader(root_dir=config.getArticleGenerationPromptsDir())
    llm_client = LiteLLMClient()

    institutional_memory = InstitutionalMemoryStore(
        data_dir=config.getArticleGenerationInstitutionalMemoryDir(),
        fact_checking_subdir=article_generation_config.institutional_memory.fact_checking_subdir,
        evidence_finding_subdir=article_generation_config.institutional_memory.evidence_finding_subdir,
    )

    output_handler = OutputHandler(
        final_articles_dir=config.getArticleGenerationOutputDir(),
        run_artifacts_dir=config.getArticleGenerationArtifactsDir(),
    )

    agents_config = article_generation_config.agents
    specialists_config = agents_config.specialists

    # --- Writer ---
    _check_agent_name(agents_config.writer.agent_name, "writer")
    if agents_config.writer.agent_name == "mock":
        writer_agent: WriterAgentProtocol = MockWriterAgent()
    else:
        writer_agent = WriterAgent(
            llm_config=agents_config.writer.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            writer_prompt_file=prompts_config.writer_prompt_file,
            revision_prompt_file=prompts_config.revision_prompt_file,
        )

    # --- Article review ---
    _check_agent_name(agents_config.article_review.agent_name, "article_review")
    if agents_config.article_review.agent_name == "mock":
        article_review_agent: ArticleReviewAgentProtocol = MockArticleReviewAgent()
    else:
        article_review_agent = ArticleReviewAgent(
            llm_config=agents_config.article_review.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            prompt_file=prompts_config.article_review_prompt_file,
        )

    # --- Concern mapping ---
    _check_agent_name(agents_config.concern_mapping.agent_name, "concern_mapping")
    if agents_config.concern_mapping.agent_name == "mock":
        concern_mapping_agent: ConcernMappingAgentProtocol = MockConcernMappingAgent()
    else:
        concern_mapping_agent = ConcernMappingAgent(
            llm_config=agents_config.concern_mapping.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            prompt_file=prompts_config.concern_mapping_prompt_file,
        )

    # --- Fact check ---
    _check_agent_name(specialists_config.fact_check.agent_name, "fact_check")
    if specialists_config.fact_check.agent_name == "mock":
        fact_check_agent: SpecialistAgentProtocol = MockFactCheckAgent()
    else:
        kb_config = article_generation_config.knowledge_base
        kb_indexer = KnowledgeBaseIndexer(
            data_dir=config.getArticleGenerationKbDir(),
            index_dir=config.getArticleGenerationKbIndexDir(),
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
            index_dir=config.getArticleGenerationKbIndexDir(),
            embedding_provider=kb_config.embedding.provider,
            embedding_model_name=kb_config.embedding.model_name,
            embedding_api_base=kb_config.embedding.api_base,
            embedding_api_key=kb_config.embedding.api_key,
            embedding_timeout_seconds=kb_config.embedding.timeout_seconds,
        )
        fact_check_agent = FactCheckAgent(
            llm_config=specialists_config.fact_check.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            specialists_dir=prompts_config.specialists_dir,
            prompt_file=prompts_config.fact_check_prompt_file,
            knowledge_base_retriever=kb_retriever,
            institutional_memory=institutional_memory,
            kb_index_version=kb_index_version,
            kb_timeout_seconds=kb_config.timeout_seconds,
        )

    # --- Evidence finding ---
    _check_agent_name(specialists_config.evidence_finding.agent_name, "evidence_finding")
    if specialists_config.evidence_finding.agent_name == "mock":
        evidence_finding_agent: SpecialistAgentProtocol = MockEvidenceFindingAgent()
    else:
        perplexity_client = PerplexityHTTPClient(
            api_base=article_generation_config.perplexity.api_base,
            api_key=article_generation_config.perplexity.api_key,
        )
        evidence_finding_agent = EvidenceFindingAgent(
            llm_config=specialists_config.evidence_finding.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            specialists_dir=prompts_config.specialists_dir,
            prompt_file=prompts_config.evidence_finding_prompt_file,
            perplexity_client=perplexity_client,
            perplexity_model=article_generation_config.perplexity.model,
            institutional_memory=institutional_memory,
        )

    # --- Opinion ---
    _check_agent_name(specialists_config.opinion.agent_name, "opinion")
    if specialists_config.opinion.agent_name == "mock":
        opinion_agent: SpecialistAgentProtocol = MockOpinionAgent()
    else:
        opinion_agent = OpinionAgent(
            llm_config=specialists_config.opinion.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            specialists_dir=prompts_config.specialists_dir,
            prompt_file=prompts_config.opinion_prompt_file,
        )

    # --- Attribution ---
    _check_agent_name(specialists_config.attribution.agent_name, "attribution")
    if specialists_config.attribution.agent_name == "mock":
        attribution_agent: SpecialistAgentProtocol = MockAttributionAgent()
    else:
        attribution_agent = AttributionAgent(
            llm_config=specialists_config.attribution.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            specialists_dir=prompts_config.specialists_dir,
            prompt_file=prompts_config.attribution_prompt_file,
        )

    # --- Style review ---
    _check_agent_name(specialists_config.style_review.agent_name, "style_review")
    if specialists_config.style_review.agent_name == "mock":
        style_review_agent: SpecialistAgentProtocol = MockStyleReviewAgent()
    else:
        style_review_agent = StyleReviewAgent(
            llm_config=specialists_config.style_review.llm,
            config=config,
            llm_client=llm_client,
            prompt_loader=prompt_loader,
            specialists_dir=prompts_config.specialists_dir,
            prompt_file=prompts_config.style_review_prompt_file,
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

"""
SQuAD Dataset Loader
---------------------
Loads a subset of SQuAD v1.1 from HuggingFace datasets and converts
each article into Documents + QA pairs ready for the CAG-RAG pipeline.

Output per article
------------------
  SQuADArticle
    .title      : str
    .documents  : List[Document]   — one Document per paragraph
    .qa_pairs   : List[QAPair]     — questions + gold answers for this article

Usage
-----
    from cag_rag.data.squad_loader import SQuADLoader

    loader = SQuADLoader(max_articles=5, max_qa_per_article=3)
    articles = loader.load()
    for article in articles:
        print(article.title, len(article.documents), "docs,", len(article.qa_pairs), "QA pairs")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..graph.builder import Document

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """One question–answer pair from SQuAD."""
    question_id: str
    question: str
    gold_answer: str          # first answer in the answer list
    context_doc_id: str       # which document (paragraph) this QA is from


@dataclass
class SQuADArticle:
    """All data for one Wikipedia article in SQuAD."""
    title: str
    documents: List[Document]    # paragraphs as Document objects
    qa_pairs: List[QAPair]


class SQuADLoader:
    """
    Loads SQuAD v1.1 validation split from HuggingFace datasets.
    Uses the validation split to avoid licensing issues and because
    it is smaller (10 570 QA pairs across 202 articles).

    Parameters
    ----------
    max_articles : int
        Number of Wikipedia articles to load. Each article has multiple
        paragraphs and multiple QA pairs.
    max_qa_per_article : int
        Cap on QA pairs per article (avoids very large articles dominating).
    split : str
        "validation" (default) or "train".
    """

    def __init__(
        self,
        max_articles: int = 10,
        max_qa_per_article: int = 5,
        split: str = "validation",
    ):
        self.max_articles = max_articles
        self.max_qa_per_article = max_qa_per_article
        self.split = split

    def load(self) -> List[SQuADArticle]:
        """
        Download (or load from cache) SQuAD and return structured articles.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed. Run: pip install datasets"
            )

        logger.info(f"Loading SQuAD v1.1 ({self.split} split) …")
        dataset = load_dataset("rajpurkar/squad", split=self.split)
        logger.info(f"  → {len(dataset)} total QA pairs loaded")

        return self._group_by_article(dataset)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _group_by_article(self, dataset) -> List[SQuADArticle]:
        """Group flat SQuAD rows by article title."""
        # title → {contexts: set, qa_pairs: list}
        article_map: dict = {}

        for row in dataset:
            title = row["title"]
            if title not in article_map:
                article_map[title] = {"contexts": {}, "qa_pairs": []}

            ctx = row["context"]
            ctx_id = f"{title}_ctx{len(article_map[title]['contexts'])}"

            # Store unique contexts (paragraphs) keyed by content
            if ctx not in article_map[title]["contexts"]:
                article_map[title]["contexts"][ctx] = ctx_id

            doc_id = article_map[title]["contexts"][ctx]

            answers = row["answers"]["text"]
            gold = answers[0] if answers else ""

            article_map[title]["qa_pairs"].append(
                QAPair(
                    question_id=row["id"],
                    question=row["question"],
                    gold_answer=gold,
                    context_doc_id=doc_id,
                )
            )

        # Convert to SQuADArticle objects, capped at max_articles
        articles: List[SQuADArticle] = []
        for title, data in list(article_map.items())[: self.max_articles]:
            documents = [
                Document(doc_id=doc_id, text=ctx, metadata={"title": title})
                for ctx, doc_id in data["contexts"].items()
            ]
            qa_pairs = data["qa_pairs"][: self.max_qa_per_article]
            articles.append(
                SQuADArticle(title=title, documents=documents, qa_pairs=qa_pairs)
            )
            logger.info(
                f"  Article '{title}': {len(documents)} paragraphs, "
                f"{len(qa_pairs)} QA pairs"
            )

        logger.info(f"Loaded {len(articles)} articles total.")
        return articles

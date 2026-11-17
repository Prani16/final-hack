"""Download the 10-paper Advanced-RAG collection and write data/catalogue.json.

    python scripts/download_papers.py

All papers are open access (arXiv, ACL Anthology). PDFs are saved to data/papers/
and are NOT committed to GitHub (see .gitignore): check each paper's licence
before redistributing. Already-downloaded files are skipped.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ingest import catalogue  # noqa: E402

PAPERS = [
    dict(paper_id="graphrag", short_name="GraphRAG", year=2024, foundational=True,
         title="From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
         authors="Edge, Trinh, Cheng, Bradley, Chao, Mody, Truitt, Larson et al.",
         source="arXiv", identifier="arXiv:2404.16130", url="https://arxiv.org/pdf/2404.16130"),
    dict(paper_id="lightrag", short_name="LightRAG", year=2024, foundational=True,
         title="LightRAG: Simple and Fast Retrieval-Augmented Generation",
         authors="Guo, Xia, Yu, Ao, Huang", source="arXiv", identifier="arXiv:2410.05779",
         url="https://arxiv.org/pdf/2410.05779"),
    dict(paper_id="visrag", short_name="VisRAG", year=2025,
         title="VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents",
         authors="Yu, Tang, Xu, Cui, Ran, Yan, Liu, Wang, Han, Liu, Sun",
         source="ICLR 2025 (OpenReview) / arXiv", identifier="arXiv:2410.10594", url="https://arxiv.org/pdf/2410.10594"),
    dict(paper_id="minirag", short_name="MiniRAG", year=2025,
         title="MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation",
         authors="Fan, Wang, Ren, Huang", source="arXiv", identifier="arXiv:2501.06713",
         url="https://arxiv.org/pdf/2501.06713"),
    dict(paper_id="agentic_rag_survey", short_name="Agentic RAG Survey", year=2025, survey=True,
         title="Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG",
         authors="Singh, Ehtesham, Kumar, Talaei Khoei, Vasilakos", source="arXiv", identifier="arXiv:2501.09136",
         url="https://arxiv.org/pdf/2501.09136"),
    dict(paper_id="pathrag", short_name="PathRAG", year=2025,
         title="PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths",
         authors="Chen, Guo, Yang, Chen, Chen, Liu, Shi, Yang", source="arXiv", identifier="arXiv:2502.14902",
         url="https://arxiv.org/pdf/2502.14902"),
    dict(paper_id="hipporag2", short_name="HippoRAG 2", year=2025,
         title="From RAG to Memory: Non-Parametric Continual Learning for Large Language Models",
         authors="Jiménez Gutiérrez, Shu, Qi, Zhou, Su", source="arXiv", identifier="arXiv:2502.14802",
         url="https://arxiv.org/pdf/2502.14802"),
    dict(paper_id="search_r1", short_name="Search-R1", year=2025,
         title="Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning",
         authors="Jin, Zeng, Yue, Yoon, Arik, Wang, Zamani, Han", source="arXiv", identifier="arXiv:2503.09516",
         url="https://arxiv.org/pdf/2503.09516"),
    dict(paper_id="setr", short_name="SetR", year=2025,
         title="Shifting from Ranking to Set Selection for Retrieval Augmented Generation",
         authors="Lee, Jo, Park, Lee", source="ACL Anthology (ACL 2025)", identifier="2025.acl-long.861",
         url="https://aclanthology.org/2025.acl-long.861.pdf"),
    dict(paper_id="rag_anything", short_name="RAG-Anything", year=2025,
         title="RAG-Anything: All-in-One RAG Framework",
         authors="Guo, Ren, Xu, Zhang, Huang", source="arXiv", identifier="arXiv:2510.12323",
         url="https://arxiv.org/pdf/2510.12323"),
]


def main():
    headers = {"User-Agent": "ResearchIQ-hackathon/1.0 (academic use)"}
    for p in PAPERS:
        dest = config.PAPERS_DIR / f"{p['paper_id']}.pdf"
        if not dest.exists():
            print(f"Downloading {p['short_name']} ...", end=" ", flush=True)
            try:
                r = requests.get(p["url"], headers=headers, timeout=60)
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    raise ValueError("response is not a PDF")
                dest.write_bytes(r.content)
                print(f"ok ({len(r.content)//1024} KB)")
            except Exception as e:
                print(f"FAILED ({e}). Download it manually from {p['url']} and save as {dest.name}")
            time.sleep(3)  # be polite to arXiv
        else:
            print(f"{p['short_name']}: already downloaded")
        catalogue.upsert(p["paper_id"], force=True, filename=dest.name, **{k: v for k, v in p.items() if k != "paper_id"})
    print(f"\nCatalogue written to {config.CATALOGUE_PATH}")


if __name__ == "__main__":
    main()

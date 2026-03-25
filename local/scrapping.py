from __future__ import annotations

import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

TOTAL_PAPERS = 3000
BATCH_SIZE = 100
REQUEST_DELAY_SECONDS = 1.0
DOWNLOAD_TIMEOUT = 60
MAX_START_OFFSET = 5000
MAX_EMPTY_ROUNDS = 30
MAX_FETCH_RETRIES = 4
MAX_DOWNLOAD_RETRIES = 3
BACKOFF_BASE_SECONDS = 2
MAX_TITLE_LENGTH = 120
REQUEST_HEADERS = {
    "User-Agent": "PaperScan/1.0 (academic project; contact: local)"
}

# Diverse queries to increase topical variety while keeping a scientific/academic focus.
SEARCH_QUERIES = [
    "cat:cs.IR",
    "cat:cs.CL",
    "cat:cs.AI",
    "cat:cs.LG",
    "cat:cs.CV",
    "cat:cs.RO",
    "cat:math.ST",
    "cat:stat.ML",
    "cat:q-bio.BM",
    "cat:q-bio.NC",
    "cat:physics.soc-ph",
    "cat:econ.EM",
    "cat:eess.SP",
    "all:scientific",
    "all:academic",
    "all:research",
    "all:information retrieval",
    "all:knowledge graph",
    "all:natural language processing",
    "all:machine learning",
]


def safe_filename(raw: str) -> str:
    compact = re.sub(r"\s+", " ", raw).strip()
    sanitized = "".join(char if char.isalnum() or char in "- ." else " " for char in compact)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized[:MAX_TITLE_LENGTH].strip()


def rename_existing_files_to_spaces(papers_dir: Path) -> None:
    for pdf_path in papers_dir.glob("*.pdf"):
        stem = pdf_path.stem
        normalized = stem

        # Legacy ID-only format, e.g. 2603_16163v1 -> 2603.16163v1
        if re.fullmatch(r"\d{4}_\d{5}v\d+", stem):
            normalized = stem.replace("_", ".", 1)
        else:
            normalized = normalized.replace("__", " ").replace("_", " ")
            normalized = re.sub(r"\s+", " ", normalized).strip()

        if normalized == stem or not normalized:
            continue

        target = pdf_path.with_name(f"{normalized}.pdf")
        if target.exists():
            continue

        pdf_path.rename(target)


def manifest_path(papers_dir: Path) -> Path:
    return papers_dir / "downloaded_ids.txt"


def load_seen_ids(papers_dir: Path) -> set[str]:
    seen: set[str] = set()
    manifest = manifest_path(papers_dir)

    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            paper_id = line.strip()
            if paper_id:
                seen.add(paper_id)

    # Backward compatibility for old files named only with arXiv ID.
    for pdf_path in papers_dir.glob("*.pdf"):
        stem = pdf_path.stem.strip()
        if re.fullmatch(r"\d{4}\.\d{5}(v\d+)?", stem):
            seen.add(stem)

    return seen


def append_seen_id(papers_dir: Path, paper_id: str) -> None:
    manifest = manifest_path(papers_dir)
    with manifest.open("a", encoding="utf-8") as file:
        file.write(f"{paper_id}\n")


def fetch_entries(search_query: str, start: int, max_results: int) -> list[ET.Element]:
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            response = requests.get(
                ARXIV_API_URL,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=DOWNLOAD_TIMEOUT,
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
            return root.findall("atom:entry", ATOM_NS)
        except (requests.RequestException, ET.ParseError) as error:
            if attempt == MAX_FETCH_RETRIES:
                print(
                    f"Fetch failed after {MAX_FETCH_RETRIES} retries "
                    f"(query={search_query}, start={start}): {error}"
                )
                return []
            backoff = BACKOFF_BASE_SECONDS ** attempt
            print(
                f"Fetch retry {attempt}/{MAX_FETCH_RETRIES} "
                f"(query={search_query}, start={start}) after error: {error}"
            )
            time.sleep(backoff)

    return []


def extract_paper_data(entry: ET.Element) -> tuple[str, str, str]:
    paper_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).split("/")[-1]
    title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()

    pdf_url = ""
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href", "")
            break

    if not pdf_url and paper_id:
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    return paper_id, title, pdf_url


def download_pdf(pdf_url: str, destination: Path) -> None:
    temp_destination = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            with requests.get(
                pdf_url,
                stream=True,
                headers=REQUEST_HEADERS,
                timeout=DOWNLOAD_TIMEOUT,
            ) as response:
                response.raise_for_status()
                with temp_destination.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
            temp_destination.replace(destination)
            return
        except requests.RequestException as error:
            if temp_destination.exists():
                temp_destination.unlink(missing_ok=True)
            if attempt == MAX_DOWNLOAD_RETRIES:
                raise error
            backoff = BACKOFF_BASE_SECONDS ** attempt
            time.sleep(backoff)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    papers_dir = base_dir / "local_papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    rename_existing_files_to_spaces(papers_dir)

    existing_pdf_count = sum(1 for _ in papers_dir.glob("*.pdf"))
    seen_ids = load_seen_ids(papers_dir)
    downloaded_now = 0
    empty_rounds = 0

    print(f"Existing PDFs: {existing_pdf_count}")
    print(f"Known paper IDs from previous runs: {len(seen_ids)}")
    if len(seen_ids) >= TOTAL_PAPERS:
        print(f"Target already reached ({len(seen_ids)} >= {TOTAL_PAPERS}).")
        print(f"Done. PDFs available in: {papers_dir}")
        return

    while len(seen_ids) < TOTAL_PAPERS and empty_rounds < MAX_EMPTY_ROUNDS:
        batch_target = min(BATCH_SIZE, TOTAL_PAPERS - len(seen_ids))
        query = random.choice(SEARCH_QUERIES)
        start = random.randint(0, MAX_START_OFFSET)
        entries = fetch_entries(search_query=query, start=start, max_results=batch_target)

        if not entries:
            empty_rounds += 1
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        random.shuffle(entries)
        added_in_round = 0

        for entry in entries:
            if len(seen_ids) >= TOTAL_PAPERS:
                break

            paper_id, title, pdf_url = extract_paper_data(entry)
            if not paper_id or not pdf_url or paper_id in seen_ids:
                continue

            safe_title = safe_filename(title) or "untitled_paper"
            output_name = f"{safe_title} {paper_id}.pdf"
            destination = papers_dir / output_name
            if destination.exists():
                seen_ids.add(paper_id)
                append_seen_id(papers_dir, paper_id)
                continue

            try:
                download_pdf(pdf_url=pdf_url, destination=destination)
                seen_ids.add(paper_id)
                append_seen_id(papers_dir, paper_id)
                downloaded_now += 1
                added_in_round += 1
                print(f"[{len(seen_ids)}/{TOTAL_PAPERS}] Downloaded {paper_id} (query: {query})")
            except requests.RequestException as error:
                print(f"Failed downloading {paper_id}: {error}")

        if added_in_round == 0:
            empty_rounds += 1
        else:
            empty_rounds = 0

        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"New PDFs downloaded in this run: {downloaded_now}")
    print(f"Total PDFs currently available: {len(seen_ids)}")
    print(f"Done. PDFs available in: {papers_dir}")


if __name__ == "__main__":
    main()

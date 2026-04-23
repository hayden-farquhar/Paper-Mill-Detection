"""
Fetch full-text XML articles from PubMed Central (PMC) Open Access.

Downloads and parses PMC XML to extract structured full text for
paper mill detection features (formulaic structure, methods reuse, etc.).
"""

import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PMC_OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
PMC_EFETCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Rate limit: NCBI allows 3 requests/second without API key, 10 with
REQUEST_DELAY = 0.35  # seconds between requests


def fetch_pmc_xml(pmcid: str, output_dir: Optional[Path] = None) -> Optional[str]:
    """Fetch full-text XML for a single PMC article.

    Args:
        pmcid: PMC identifier (e.g., "PMC1234567")
        output_dir: If provided, save XML to this directory

    Returns:
        XML string, or None if not available
    """
    # Normalise PMCID format
    pmcid_clean = pmcid.replace("https://www.ncbi.nlm.nih.gov/pmc/articles/", "")
    pmcid_clean = pmcid_clean.strip("/")
    if not pmcid_clean.startswith("PMC"):
        pmcid_clean = f"PMC{pmcid_clean}"

    # Use E-utilities efetch for PMC XML
    params = {
        "db": "pmc",
        "id": pmcid_clean.replace("PMC", ""),
        "rettype": "xml",
        "retmode": "xml",
    }

    try:
        resp = requests.get(PMC_EFETCH_BASE, params=params, timeout=30)
        resp.raise_for_status()

        xml_text = resp.text
        if "<pmc-articleset" not in xml_text and "<article" not in xml_text:
            logger.warning(f"No article XML returned for {pmcid_clean}")
            return None

        # Save if output dir specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            xml_path = output_dir / f"{pmcid_clean}.xml"
            xml_path.write_text(xml_text, encoding="utf-8")
            logger.debug(f"Saved {xml_path}")

        return xml_text

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {pmcid_clean}: {e}")
        return None


def parse_pmc_xml(xml_text: str) -> dict:
    """Parse PMC XML into structured sections.

    Returns a dict with keys: title, abstract, sections (list of
    {heading, text} dicts), references_count, and raw body text.
    """
    result = {
        "title": "",
        "abstract": "",
        "sections": [],
        "body_text": "",
        "references_count": 0,
        "section_headings": [],
    }

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"XML parse error: {e}")
        return result

    # Find the article element (may be nested in pmc-articleset)
    article = root.find(".//article")
    if article is None:
        article = root  # The root might be the article itself

    # Title
    title_el = article.find(".//article-title")
    if title_el is not None:
        result["title"] = _get_all_text(title_el)

    # Abstract
    abstract_el = article.find(".//abstract")
    if abstract_el is not None:
        result["abstract"] = _get_all_text(abstract_el)

    # Body sections
    body = article.find(".//body")
    if body is not None:
        sections = []
        for sec in body.findall(".//sec"):
            title_el = sec.find("title")
            heading = _get_all_text(title_el) if title_el is not None else ""

            # Get direct paragraph text (not from child sections)
            paragraphs = []
            for p in sec.findall("p"):
                paragraphs.append(_get_all_text(p))

            if paragraphs:
                section_text = " ".join(paragraphs)
                sections.append({"heading": heading, "text": section_text})

        result["sections"] = sections
        result["section_headings"] = [s["heading"] for s in sections if s["heading"]]
        result["body_text"] = " ".join(s["text"] for s in sections)

    # Reference count
    ref_list = article.find(".//ref-list")
    if ref_list is not None:
        result["references_count"] = len(ref_list.findall("ref"))

    return result


def _get_all_text(element: ET.Element) -> str:
    """Extract all text content from an XML element, including tail text."""
    return " ".join(element.itertext()).strip()


def fetch_batch(
    pmcids: list[str],
    output_dir: Optional[Path] = None,
    parse: bool = True,
) -> list[dict]:
    """Fetch and optionally parse multiple PMC articles.

    Args:
        pmcids: List of PMC IDs
        output_dir: Directory to cache downloaded XML
        parse: If True, parse XML into structured data

    Returns:
        List of parsed article dicts (if parse=True) or raw XML strings
    """
    results = []

    for i, pmcid in enumerate(pmcids):
        if i > 0:
            time.sleep(REQUEST_DELAY)

        if i % 10 == 0:
            logger.info(f"Fetching PMC articles: {i}/{len(pmcids)}")

        xml_text = fetch_pmc_xml(pmcid, output_dir=output_dir)
        if xml_text is None:
            continue

        if parse:
            parsed = parse_pmc_xml(xml_text)
            parsed["pmcid"] = pmcid
            results.append(parsed)
        else:
            results.append({"pmcid": pmcid, "xml": xml_text})

    logger.info(f"Successfully fetched {len(results)}/{len(pmcids)} articles")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with a known open-access medical AI paper
    test_pmcid = "PMC6945006"  # Example PMC article
    print(f"Fetching {test_pmcid}...")

    xml = fetch_pmc_xml(test_pmcid)
    if xml:
        parsed = parse_pmc_xml(xml)
        print(f"Title: {parsed['title'][:100]}")
        print(f"Sections: {parsed['section_headings']}")
        print(f"Body length: {len(parsed['body_text'])} chars")
        print(f"References: {parsed['references_count']}")
    else:
        print("Could not fetch article")

"""Fast, no-network, no-model unit tests for the ingestion pipeline logic."""

from pathlib import Path

from visawise.ingestion.chunk import (
    Chunk,
    corpus_hash,
    make_piece_id,
    read_markdown_with_frontmatter,
)
from visawise.ingestion.extract import strip_trailing_link_sections


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="d", seq=0, text="t", url="u",
        title="T", topic="x", section="", fetched_at="2026-07-08",
    )


class TestStripTrailingLinkSections:
    def test_drops_empty_link_box(self):
        body = "# Page\n\nReal prose here.\n\n#### Related Links\n\n**Form**\n\n**Other USCIS Links**"
        assert strip_trailing_link_sections(body) == "# Page\n\nReal prose here."

    def test_drops_anchor_text_list(self):
        body = "Prose.\n\n#### More Information\n\n- H-1B Cap Season\n- Registration Process"
        assert strip_trailing_link_sections(body) == "Prose."

    def test_keeps_section_with_real_prose(self):
        prose = "This paragraph explains something substantive about eligibility " * 5
        body = f"Intro.\n\n#### More Information\n\n{prose}"
        assert strip_trailing_link_sections(body) == body

    def test_no_heading_is_noop(self):
        body = "Just content, for more information see your DSO."
        assert strip_trailing_link_sections(body) == body


class TestFrontmatter:
    def test_roundtrip(self, tmp_path: Path):
        md = tmp_path / "doc.md"
        md.write_text("---\ndoc_id: abc\ntitle: T\n---\n\nBody text.\n")
        metadata, body = read_markdown_with_frontmatter(md)
        assert metadata == {"doc_id": "abc", "title": "T"}
        assert body == "Body text."

    def test_missing_frontmatter_raises(self, tmp_path: Path):
        md = tmp_path / "doc.md"
        md.write_text("No frontmatter here.")
        try:
            read_markdown_with_frontmatter(md)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


class TestDeterminism:
    def test_piece_id_stable_and_distinct(self):
        a = make_piece_id("https://x.gov/a", 0, "hello world")
        assert a == make_piece_id("https://x.gov/a", 0, "hello world")
        assert len(a) == 24
        assert a != make_piece_id("https://x.gov/a", 1, "hello world")
        assert a != make_piece_id("https://x.gov/a", 0, "hello world!")

    def test_corpus_hash_order_independent(self):
        one = [make_chunk("a"), make_chunk("b")]
        two = [make_chunk("b"), make_chunk("a")]
        assert corpus_hash(one) == corpus_hash(two)
        assert corpus_hash(one) != corpus_hash([make_chunk("a")])

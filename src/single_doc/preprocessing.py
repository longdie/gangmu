from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.common.files import clean_whitespace, read_text


MAX_HEADING_LEVEL = 6
PAGE_MARKER_PATTERN = re.compile(
    r"<!--\s*page\s+0*(\d+)\s*-->",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Segment:
    segment_id: str
    title_path: list[str]
    text: str
    token_count: int
    pages: list[int]
    next_heading_level: int = 1

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "title_path": list(self.title_path),
            "text": self.text,
            "token_count": self.token_count,
            "pages": list(self.pages),
            "next_heading_level": self.next_heading_level,
        }


class MarkdownPreprocessor:
    """Split OCR Markdown without textbook-only recall or repair branches."""

    def __init__(
        self,
        *,
        max_tokens: int,
        retry_overlap_tokens: int = 1000,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        if retry_overlap_tokens < 0:
            raise ValueError("retry_overlap_tokens cannot be negative.")
        self.max_tokens = max_tokens
        self.retry_overlap_tokens = retry_overlap_tokens
        self._encoding = None

    def segment_file(self, markdown_path: str | Path) -> list[dict]:
        return self.segment_markdown(self.clean_markdown(read_text(markdown_path)))

    def clean_markdown(self, markdown: str) -> str:
        code_blocks: list[str] = []

        def preserve_code_block(match: re.Match[str]) -> str:
            code_blocks.append(match.group(0))
            return f"__GANGMU_CODE_BLOCK_{len(code_blocks) - 1}__"

        cleaned = re.sub(r"```[\s\S]*?```", preserve_code_block, markdown or "")
        cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        def restore_code_block(match: re.Match[str]) -> str:
            return code_blocks[int(match.group(1))]

        cleaned = re.sub(
            r"__GANGMU_CODE_BLOCK_(\d+)__",
            restore_code_block,
            cleaned,
        )
        return cleaned.strip()

    def segment_markdown(self, markdown: str) -> list[dict]:
        raw_segments = self._recursive_split(
            markdown.strip(),
            title_path=[],
            heading_level=1,
        )
        segments: list[dict] = []
        current_page: int | None = None
        for title_path, text, next_heading_level in raw_segments:
            text = text.strip()
            if not text:
                continue
            pages = self.extract_pages(text)
            if pages:
                current_page = pages[-1]
            elif current_page is not None:
                pages = [current_page]
            segment = Segment(
                segment_id=f"seg_{len(segments) + 1:04d}",
                title_path=title_path,
                text=text,
                token_count=self.count_tokens(text),
                pages=pages,
                next_heading_level=next_heading_level,
            )
            segments.append(segment.to_dict())
        return segments

    def split_segment_for_retry(self, segment: dict) -> list[dict]:
        text = str(segment.get("text", "") or "").strip()
        if not text:
            return []

        start_heading_level = self._retry_start_heading_level(segment)
        for heading_level in range(start_heading_level, MAX_HEADING_LEVEL + 1):
            sections = self._split_by_heading(text, heading_level)
            if len(sections) > 1:
                return self._make_child_segments(
                    parent=segment,
                    pieces=sections,
                    suffix=f"h{heading_level}",
                    next_heading_level=heading_level + 1,
                    method=f"heading_{heading_level}",
                )

        paragraphs = self._split_paragraphs(text)
        if len(paragraphs) > 1:
            midpoint = max(1, len(paragraphs) // 2)
            pieces = [
                ("", "\n\n".join(paragraphs[:midpoint])),
                ("", "\n\n".join(paragraphs[midpoint:])),
            ]
            return self._make_child_segments(
                parent=segment,
                pieces=pieces,
                suffix="paragraph",
                next_heading_level=MAX_HEADING_LEVEL + 1,
                method="paragraph_halves",
            )

        token_count = self.count_tokens(text)
        if token_count <= 1:
            return []
        chunk_size = max(1, min(self.max_tokens, token_count // 2))
        overlap = min(self.retry_overlap_tokens, max(0, chunk_size // 5))
        token_pieces = [
            ("", piece)
            for piece in self.split_text_by_tokens(
                text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        ]
        return self._make_child_segments(
            parent=segment,
            pieces=token_pieces,
            suffix="window",
            next_heading_level=MAX_HEADING_LEVEL + 1,
            method="sliding_window",
        )

    def count_tokens(self, text: str) -> int:
        try:
            if self._encoding is None:
                import tiktoken

                self._encoding = tiktoken.get_encoding("cl100k_base")
            return len(self._encoding.encode(text or ""))
        except Exception:
            return max(1, len(text or "") // 4)

    def split_text_by_tokens(
        self,
        text: str,
        *,
        chunk_size: int,
        overlap: int = 0,
    ) -> list[str]:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if overlap < 0:
            raise ValueError("overlap cannot be negative.")
        if overlap >= chunk_size:
            overlap = 0

        try:
            if self._encoding is None:
                import tiktoken

                self._encoding = tiktoken.get_encoding("cl100k_base")
            tokens = self._encoding.encode(text or "")
            chunks: list[str] = []
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunks.append(self._encoding.decode(tokens[start:end]))
                if end >= len(tokens):
                    break
                start = end - overlap
            return chunks
        except Exception:
            chunk_chars = max(1, chunk_size * 4)
            overlap_chars = max(0, overlap * 4)
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_chars, len(text))
                chunks.append(text[start:end])
                if end >= len(text):
                    break
                start = end - overlap_chars
            return chunks

    def extract_pages(self, text: str) -> list[int]:
        return list(
            dict.fromkeys(
                int(match)
                for match in PAGE_MARKER_PATTERN.findall(text or "")
                if int(match) > 0
            )
        )

    def _recursive_split(
        self,
        text: str,
        *,
        title_path: list[str],
        heading_level: int,
    ) -> list[tuple[list[str], str, int]]:
        if self.count_tokens(text) <= self.max_tokens:
            return [(title_path, text, heading_level)]

        if heading_level <= MAX_HEADING_LEVEL:
            sections = self._split_by_heading(text, heading_level)
            if len(sections) > 1:
                result: list[tuple[list[str], str, int]] = []
                for title, section_text in sections:
                    next_path = title_path + ([title] if title else [])
                    result.extend(
                        self._recursive_split(
                            section_text,
                            title_path=next_path,
                            heading_level=heading_level + 1,
                        )
                    )
                return result
            return self._recursive_split(
                text,
                title_path=title_path,
                heading_level=heading_level + 1,
            )

        chunks = self._pack_paragraphs(text)
        return [
            (title_path, chunk, MAX_HEADING_LEVEL + 1)
            for chunk in chunks
            if chunk.strip()
        ]

    def _pack_paragraphs(self, text: str) -> list[str]:
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            if paragraph_tokens > self.max_tokens:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                chunks.extend(
                    self.split_text_by_tokens(
                        paragraph,
                        chunk_size=self.max_tokens,
                        overlap=0,
                    )
                )
                continue
            if current and current_tokens + paragraph_tokens > self.max_tokens:
                chunks.append("\n\n".join(current))
                current = [paragraph]
                current_tokens = paragraph_tokens
            else:
                current.append(paragraph)
                current_tokens += paragraph_tokens
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _make_child_segments(
        self,
        *,
        parent: dict,
        pieces: list[tuple[str, str]],
        suffix: str,
        next_heading_level: int,
        method: str,
    ) -> list[dict]:
        result: list[dict] = []
        parent_title_path = list(parent.get("title_path", []))
        parent_pages = list(parent.get("pages", []))
        current_page = parent_pages[0] if parent_pages else None
        for index, (title, piece) in enumerate(pieces, start=1):
            piece = piece.strip()
            if not piece:
                continue
            pages = self.extract_pages(piece)
            if pages:
                current_page = pages[-1]
            elif current_page is not None:
                pages = [current_page]
            result.append(
                {
                    "segment_id": (
                        f"{parent.get('segment_id', 'seg')}_{suffix}_{index}"
                    ),
                    "title_path": parent_title_path + ([title] if title else []),
                    "text": piece,
                    "token_count": self.count_tokens(piece),
                    "pages": pages,
                    "next_heading_level": next_heading_level,
                    "split_method": method,
                }
            )
        return result

    def _retry_start_heading_level(self, segment: dict) -> int:
        try:
            level = int(segment.get("next_heading_level", 1))
        except (TypeError, ValueError):
            return 1
        return max(1, min(level, MAX_HEADING_LEVEL + 1))

    def _split_by_heading(
        self,
        text: str,
        heading_level: int,
    ) -> list[tuple[str, str]]:
        pattern = re.compile(rf"(?m)^(#{{{heading_level}}}\s+.+)$")
        matches = list(pattern.finditer(text))
        if not matches:
            return []

        sections: list[tuple[str, str]] = []
        if matches[0].start() > 0:
            prefix = text[: matches[0].start()].strip()
            if prefix:
                sections.append(("", prefix))
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            heading = clean_whitespace(match.group(1).lstrip("#").strip())
            sections.append((heading, text[start:end].strip()))
        return sections

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        return [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text or "")
            if paragraph.strip()
        ]

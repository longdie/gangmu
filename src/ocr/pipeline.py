from __future__ import annotations

import argparse
import base64
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image
from pypdf import PdfReader

from src.common.config import GangmuConfig, get_config
from src.common.files import (
    ProjectLogger,
    clean_whitespace,
    ensure_dir,
    read_json,
    read_text,
    utc_now,
    write_json,
    write_text,
)
from src.common.paths import DATA_DIR, OCR_OUTPUT_DIR, PROMPT_DIR


DEFAULT_OCR_PROMPT_PATH = PROMPT_DIR / "ocr_math_markdown.md"
OCR_COLLECTIONS = {"textbooks", "papers"}


@dataclass
class OCRRunConfig:
    pdf_path: Path
    output_dir: Path
    method: str = "premium"
    resume: bool = False
    resume_from_page: int | None = None
    first_page: int = 1
    last_page: int | None = None
    render_dpi: int = 200
    jpeg_quality: int = 95
    parallel_workers: int = 5
    page_retry_attempts: int = 3
    retry_delay_seconds: float = 3.0
    glm46v_model: str = "glm-4.6v"
    glm5v_model: str = "glm-5v"
    fast_provider: str = "ocr"
    premium_provider: str = "ocr"
    provider_api_key: str | None = field(default=None, repr=False)
    provider_base_url: str | None = None
    ocr_prompt_path: Path = DEFAULT_OCR_PROMPT_PATH
    glm46v_max_tokens: int = 16384
    glm5v_max_tokens: int = 16384
    glm46v_timeout_seconds: float = 180.0
    toc_pages: tuple[int, ...] = ()
    glm5v_retry_delay_seconds: float = 3.0
    glm5v_call_delay_seconds: float = 0.0
    document_id: str | None = None
    app_config: GangmuConfig | None = None

    @classmethod
    def from_config(cls, app_config: GangmuConfig, **overrides: Any) -> "OCRRunConfig":
        defaults = app_config.ocr()
        fast = app_config.model("ocr_fast")
        premium = app_config.model("ocr_premium")
        settings = {
            "method": defaults.method,
            "render_dpi": defaults.render_dpi,
            "jpeg_quality": defaults.jpeg_quality,
            "parallel_workers": defaults.parallel_workers,
            "page_retry_attempts": defaults.page_retry_attempts,
            "retry_delay_seconds": defaults.retry_delay_seconds,
            "glm46v_model": fast.name,
            "glm5v_model": premium.name,
            "fast_provider": fast.provider,
            "premium_provider": premium.provider,
            "glm46v_max_tokens": defaults.fast_max_tokens,
            "glm5v_max_tokens": defaults.premium_max_tokens,
            "glm46v_timeout_seconds": defaults.timeout_seconds,
            "toc_pages": defaults.toc_pages,
            "glm5v_retry_delay_seconds": defaults.premium_retry_delay_seconds,
            "glm5v_call_delay_seconds": defaults.premium_call_delay_seconds,
            "app_config": app_config,
        }
        settings.update(overrides)
        return cls(**settings)


class OCRQualityChecker:
    EN_KEYWORDS = [
        "Definition",
        "Theorem",
        "Lemma",
        "Proposition",
        "Corollary",
        "Proof",
    ]
    ZH_KEYWORDS = ["定义", "定理", "引理", "命题", "推论", "证明"]
    GARBLED_PATTERN = re.compile(r"[�]")
    BOX_SYMBOL_PATTERN = re.compile(r"[□■]")
    OCR_FAILURE_PATTERN = re.compile(
        r"\[(?:OCR failed|missing OCR page)[^\]]*\]", flags=re.IGNORECASE
    )
    CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    LATIN_PATTERN = re.compile(r"[A-Za-z]")
    SPACED_LATEX_TOKEN_PATTERN = re.compile(r"(?<!\\)[_^]\s+\{")
    SPACED_LATEX_COMMAND_PATTERN = re.compile(
        r"\\(?:mathrm|mathbf|mathbb|mathcal|mathfrak|operatorname|text)\s+\{"
    )
    SPLIT_WORD_IN_MATH_PATTERN = re.compile(
        r"\\(?:mathrm|text)\s*\{\s*[A-Za-z](?:\s+[A-Za-z]){2,}\s*\}"
    )
    POSSIBLE_UNWRAPPED_MATH_PATTERN = re.compile(
        r"\bLet\s+[a-z]\s+and\s+[a-z]\b|\bfor all\s+[a-z]\s*,\s*[a-z]\b",
        flags=re.IGNORECASE,
    )

    def check(self, markdown: str, page_count: int | None = None) -> dict[str, Any]:
        text = markdown or ""
        lines = [line.strip() for line in text.splitlines()]
        nonempty_lines = [line for line in lines if line]

        dollar_count = text.count("$")
        display_math_count = text.count("$$")
        begin_envs = re.findall(r"\\begin\{([^}]+)\}", text)
        end_envs = re.findall(r"\\end\{([^}]+)\}", text)
        garbled_count = len(self.GARBLED_PATTERN.findall(text))
        box_symbol_count = len(self.BOX_SYMBOL_PATTERN.findall(text))
        ocr_failure_count = len(self.OCR_FAILURE_PATTERN.findall(text))
        cjk_count = len(self.CJK_PATTERN.findall(text))
        latin_count = len(self.LATIN_PATTERN.findall(text))
        image_count = len(re.findall(r"!\[[^\]]*\]\([^)]+\)", text))
        html_table_count = len(re.findall(r"<table[\s>]", text, flags=re.IGNORECASE))
        md_table_line_count = sum(
            1 for line in nonempty_lines if line.startswith("|") and line.endswith("|")
        )
        headings = re.findall(r"(?m)^(#{1,6})\s+(.+)$", text)
        repeated_lines = self._find_repeated_lines(nonempty_lines)
        spaced_word_candidates = self._find_spaced_word_lines(nonempty_lines)
        spaced_latex_token_count = len(self.SPACED_LATEX_TOKEN_PATTERN.findall(text))
        spaced_latex_command_count = len(
            self.SPACED_LATEX_COMMAND_PATTERN.findall(text)
        )
        split_word_in_math_count = len(self.SPLIT_WORD_IN_MATH_PATTERN.findall(text))
        possible_unwrapped_math_count = len(
            self.POSSIBLE_UNWRAPPED_MATH_PATTERN.findall(text)
        )

        en_keyword_counts = {
            keyword: len(
                re.findall(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE)
            )
            for keyword in self.EN_KEYWORDS
        }
        zh_keyword_counts = {
            keyword: text.count(keyword) for keyword in self.ZH_KEYWORDS
        }

        char_count = len(text)
        chars_per_page = char_count / page_count if page_count else None

        issues = []
        if not text.strip():
            issues.append("markdown_empty")
        if page_count and chars_per_page is not None and chars_per_page < 300:
            issues.append("very_low_chars_per_page")
        if dollar_count % 2 != 0:
            issues.append("unbalanced_dollar_count")
        if len(begin_envs) != len(end_envs) or sorted(begin_envs) != sorted(end_envs):
            issues.append("latex_begin_end_mismatch")
        if garbled_count >= 10:
            issues.append("many_garbled_characters")
        if page_count and box_symbol_count > page_count * 5:
            issues.append("many_box_symbols")
        if self._has_hard_repetition(repeated_lines, page_count=page_count):
            issues.append("repeated_line_candidates")
        if ocr_failure_count:
            issues.append("ocr_failure_markers")
        if not headings and page_count and page_count > 1:
            issues.append("no_markdown_headings")
        if (
            page_count
            and page_count > 1
            and sum(en_keyword_counts.values()) + sum(zh_keyword_counts.values()) == 0
        ):
            issues.append("no_math_statement_keywords")
        if spaced_word_candidates:
            issues.append("spaced_word_candidates")
        if spaced_latex_token_count > max(10, (page_count or 1) // 2):
            issues.append("many_latex_internal_spaces")
        if spaced_latex_command_count > max(5, (page_count or 1) // 4):
            issues.append("many_spaced_latex_commands")
        if split_word_in_math_count > 0:
            issues.append("split_words_inside_math")
        if possible_unwrapped_math_count > 0:
            issues.append("possible_unwrapped_inline_math")

        return {
            "created_at": utc_now(),
            "page_count": page_count,
            "char_count": char_count,
            "line_count": len(lines),
            "nonempty_line_count": len(nonempty_lines),
            "chars_per_page": chars_per_page,
            "language_counts": {
                "cjk_chars": cjk_count,
                "latin_letters": latin_count,
            },
            "math_format_counts": {
                "dollar_count": dollar_count,
                "display_math_delimiter_count": display_math_count,
                "latex_begin_count": len(begin_envs),
                "latex_end_count": len(end_envs),
            },
            "structure_counts": {
                "heading_count": len(headings),
                "image_count": image_count,
                "html_table_count": html_table_count,
                "markdown_table_line_count": md_table_line_count,
            },
            "keyword_counts": {
                "english": en_keyword_counts,
                "chinese": zh_keyword_counts,
            },
            "garbled_count": garbled_count,
            "box_symbol_count": box_symbol_count,
            "ocr_failure_count": ocr_failure_count,
            "repeated_line_candidates": repeated_lines[:50],
            "spaced_word_candidates": spaced_word_candidates[:50],
            "latex_spacing_counts": {
                "spaced_latex_token_count": spaced_latex_token_count,
                "spaced_latex_command_count": spaced_latex_command_count,
                "split_word_in_math_count": split_word_in_math_count,
                "possible_unwrapped_math_count": possible_unwrapped_math_count,
            },
            "issues": issues,
        }

    def check_page_basic(self, markdown: str) -> dict[str, Any]:
        report = self.check(markdown, page_count=1)
        critical_issues = [
            issue
            for issue in report["issues"]
            if issue
            in {
                "markdown_empty",
                "very_low_chars_per_page",
                "unbalanced_dollar_count",
                "latex_begin_end_mismatch",
                "many_garbled_characters",
                "many_box_symbols",
                "ocr_failure_markers",
            }
        ]
        if self._has_hard_repetition(report["repeated_line_candidates"], page_count=1):
            critical_issues.append("repeated_line_candidates")
        report["critical_issues"] = critical_issues
        report["passed_basic_check"] = not critical_issues
        return report

    def _find_repeated_lines(
        self, lines: list[str], min_repeats: int = 4
    ) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for line in lines:
            normalized = re.sub(r"\s+", " ", line).strip()
            if len(normalized) < 5:
                continue
            if normalized.startswith("!") or normalized.startswith("|"):
                continue
            counts[normalized] = counts.get(normalized, 0) + 1
        repeated = []
        for line, count in sorted(
            counts.items(), key=lambda item: item[1], reverse=True
        ):
            threshold = min_repeats
            if line in {r"\hline", r"$\square$", r"\square", r"$\Box$", r"\Box"}:
                continue
            if re.fullmatch(r"(?:\$\$)?\\(?:begin|end)\{[^}]+\}(?:\$\$)?", line):
                threshold = 20
            if count >= threshold:
                repeated.append({"line": line, "count": count})
        return repeated

    def _has_hard_repetition(
        self, repeated_lines: list[dict[str, Any]], page_count: int | None = None
    ) -> bool:
        page_count = max(1, int(page_count or 1))
        env_threshold = max(20, page_count // 4)
        noisy_math_threshold = max(30, page_count // 3)
        for item in repeated_lines:
            line = str(item.get("line") or "")
            count = int(item.get("count") or 0)
            if count >= env_threshold and re.search(r"\\(?:begin|end)\{[^}]+\}", line):
                return True
            special_chars = len(re.findall(r"[\\{}_^$&]", line))
            alpha_chars = len(re.findall(r"[A-Za-z]", line))
            if (
                count >= noisy_math_threshold
                and special_chars >= 5
                and alpha_chars < max(5, len(line) // 5)
            ):
                return True
        return False

    def _find_spaced_word_lines(self, lines: list[str]) -> list[dict[str, Any]]:
        candidates = []
        pattern = re.compile(r"(?<![A-Za-z'])\b(?:[A-Za-z]\s+){3,}[A-Za-z]{2,}\b")
        for line_number, line in enumerate(lines, start=1):
            matches = pattern.findall(line)
            if matches:
                candidates.append(
                    {
                        "line_number": line_number,
                        "line": line,
                        "matches": matches[:10],
                    }
                )
        return candidates


class OCRMarkdownStructureNormalizer:
    """Conservative cleanup for OCR Markdown structure.

    This is intentionally narrow: it fixes common textbook section headings
    without trying to infer arbitrary document structure.
    """

    SECTION_CODE_PATTERN = re.compile(r"^(\d{1,3}[A-Z])\s+(.{2,120})$")
    EXERCISES_PATTERN = re.compile(r"^(Exercises\s+\d{1,3}[A-Z])$", flags=re.IGNORECASE)
    PAGE_MARKER_PATTERN = re.compile(r"^<!--\s*page\s+\d+\s*-->$", flags=re.IGNORECASE)
    CHAPTER_STANDALONE_PATTERN = re.compile(
        r"^Chapter\s+(\d{1,3})\s*$", flags=re.IGNORECASE
    )
    CHAPTER_TITLE_PATTERN = re.compile(
        r"^Chapter\s+(\d{1,3})\s+(.{2,120})$", flags=re.IGNORECASE
    )
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
    PAGE_HEADER_SECTION_PATTERN = re.compile(r"^Section\s+\d{1,3}[A-Z]\s+.+\s+\d{1,4}$")
    RUNNING_FOOTER_PATTERN = re.compile(
        r"^[A-Z][^\n]{5,120},\s+[^\n]{0,40}edition,\s+by\s+.+$",
        flags=re.IGNORECASE,
    )
    RESULT_HEADING_PATTERN = re.compile(
        r"^(#{1,6})\s+((?:\d+\.)+\d+\s+"
        r"(?:definition|notation|lemma|theorem|proposition|corollary|property|example)\b.*)$",
        flags=re.IGNORECASE,
    )

    def normalize(self, markdown: str) -> tuple[str, dict[str, Any]]:
        lines = (markdown or "").splitlines()
        repaired: list[str] = []
        stats = {
            "section_headings_added": 0,
            "exercise_headings_added": 0,
            "chapter_headings_added": 0,
            "heading_levels_adjusted": 0,
            "nonstructural_headings_demoted": 0,
            "result_headings_demoted": 0,
            "running_headers_removed": 0,
        }

        current_chapter_heading = ""
        current_page_is_toc = False
        seen_content_on_page = False
        skip_line_indices: set[int] = set()

        for index, line in enumerate(lines):
            if index in skip_line_indices:
                continue
            stripped = line.strip()
            if not stripped:
                repaired.append(line)
                continue

            if self.PAGE_MARKER_PATTERN.match(stripped):
                current_page_is_toc = self._is_probably_toc_page(lines, index)
                seen_content_on_page = False
                repaired.append(line)
                continue

            plain_line = self._strip_heading_marker(stripped)
            is_first_content_on_page = not seen_content_on_page

            if self._is_running_header_or_footer(plain_line):
                stats["running_headers_removed"] += 1
                continue

            result_heading = self.RESULT_HEADING_PATTERN.match(stripped)
            if result_heading:
                repaired.append(f"**{result_heading.group(2).strip()}**")
                stats["result_headings_demoted"] += 1
                seen_content_on_page = True
                continue

            heading_match = self.HEADING_PATTERN.match(stripped)
            if heading_match:
                heading_level = heading_match.group(1)
                heading_body = heading_match.group(2).strip()
                normalized_heading_body = self._normalize_heading_key(heading_body)

                heading_standalone_chapter = self.CHAPTER_STANDALONE_PATTERN.match(
                    heading_body
                )
                if (
                    heading_standalone_chapter
                    and is_first_content_on_page
                    and not current_page_is_toc
                ):
                    next_index, next_line = self._next_nonempty_plain_line(
                        lines, index + 1
                    )
                    if (
                        next_index is not None
                        and self._is_probable_chapter_title_fragment(next_line)
                    ):
                        chapter_title = f"Chapter {heading_standalone_chapter.group(1)} {next_line.strip()}"
                        current_chapter_heading = self._normalize_heading_key(
                            chapter_title
                        )
                        skip_line_indices.add(next_index)
                        repaired.append(f"# {chapter_title}")
                        stats["chapter_headings_added"] += 1
                        if heading_level != "#":
                            stats["heading_levels_adjusted"] += 1
                        seen_content_on_page = True
                        continue

                if (
                    is_first_content_on_page
                    and self.CHAPTER_TITLE_PATTERN.match(heading_body)
                    and normalized_heading_body == current_chapter_heading
                ):
                    stats["running_headers_removed"] += 1
                    continue

                if self.CHAPTER_TITLE_PATTERN.match(heading_body):
                    clean_heading = clean_whitespace(heading_body)
                    current_chapter_heading = self._normalize_heading_key(clean_heading)
                    repaired.append(f"# {clean_heading}")
                    if heading_level != "#":
                        stats["heading_levels_adjusted"] += 1
                    seen_content_on_page = True
                    continue

                exercises_heading_match = self.EXERCISES_PATTERN.match(heading_body)
                if exercises_heading_match:
                    repaired.append(f"### {exercises_heading_match.group(1)}")
                    stats["heading_levels_adjusted"] += 1
                    seen_content_on_page = True
                    continue

                section_heading_match = self.SECTION_CODE_PATTERN.match(heading_body)
                if section_heading_match and self._is_probable_section_heading(
                    section_heading_match.group(1), section_heading_match.group(2)
                ):
                    repaired.append(
                        f"## {section_heading_match.group(1)} {section_heading_match.group(2).strip()}"
                    )
                    stats["heading_levels_adjusted"] += 1
                    seen_content_on_page = True
                    continue

                if self._is_likely_nonstructural_heading(heading_body):
                    repaired.append(f"**{heading_body}**")
                    stats["nonstructural_headings_demoted"] += 1
                    seen_content_on_page = True
                    continue

                if heading_level == "#" and current_chapter_heading:
                    repaired.append(f"### {heading_body}")
                    stats["heading_levels_adjusted"] += 1
                    seen_content_on_page = True
                    continue

                repaired.append(line)
                seen_content_on_page = True
                continue

            chapter_title = self._make_chapter_heading(
                lines=lines,
                index=index,
                stripped=stripped,
                is_first_content_on_page=is_first_content_on_page,
                current_page_is_toc=current_page_is_toc,
                current_chapter_heading=current_chapter_heading,
                skip_line_indices=skip_line_indices,
            )
            if chapter_title:
                current_chapter_heading = self._normalize_heading_key(chapter_title)
                repaired.append(f"# {chapter_title}")
                stats["chapter_headings_added"] += 1
                seen_content_on_page = True
                continue

            if (
                is_first_content_on_page
                and self.CHAPTER_TITLE_PATTERN.match(stripped)
                and self._normalize_heading_key(stripped) == current_chapter_heading
            ):
                stats["running_headers_removed"] += 1
                continue

            exercises_match = self.EXERCISES_PATTERN.match(stripped)
            if exercises_match:
                repaired.append(f"### {exercises_match.group(1)}")
                stats["exercise_headings_added"] += 1
                seen_content_on_page = True
                continue

            section_match = self.SECTION_CODE_PATTERN.match(stripped)
            if section_match and self._is_probable_section_heading(
                section_match.group(1), section_match.group(2)
            ):
                repaired.append(
                    f"## {section_match.group(1)} {section_match.group(2).strip()}"
                )
                stats["section_headings_added"] += 1
                seen_content_on_page = True
                continue

            repaired.append(line)
            seen_content_on_page = True

        text = "\n".join(repaired)
        text = re.sub(r"\n{4,}", "\n\n\n", text).strip() + "\n"
        stats["created_at"] = utc_now()
        return text, stats

    def _strip_heading_marker(self, line: str) -> str:
        match = self.HEADING_PATTERN.match(line.strip())
        if match:
            return match.group(2).strip()
        return line.strip()

    def _is_running_header_or_footer(self, line: str) -> bool:
        if self.PAGE_HEADER_SECTION_PATTERN.match(line):
            return True
        if self.RUNNING_FOOTER_PATTERN.match(line):
            return True
        return False

    def _make_chapter_heading(
        self,
        lines: list[str],
        index: int,
        stripped: str,
        is_first_content_on_page: bool,
        current_page_is_toc: bool,
        current_chapter_heading: str,
        skip_line_indices: set[int],
    ) -> str:
        if not is_first_content_on_page or current_page_is_toc:
            return ""

        standalone_match = self.CHAPTER_STANDALONE_PATTERN.match(stripped)
        if standalone_match:
            next_index, next_line = self._next_nonempty_plain_line(lines, index + 1)
            if next_index is None or not self._is_probable_chapter_title_fragment(
                next_line
            ):
                return ""
            chapter_title = f"Chapter {standalone_match.group(1)} {next_line.strip()}"
            skip_line_indices.add(next_index)
            return chapter_title

        title_match = self.CHAPTER_TITLE_PATTERN.match(stripped)
        if not title_match:
            return ""
        normalized = self._normalize_heading_key(stripped)
        if normalized == current_chapter_heading:
            return ""
        _, next_line = self._next_nonempty_plain_line(lines, index + 1)
        if self._looks_like_content_after_running_chapter_header(next_line):
            return ""
        if re.search(r"\s+\d{1,4}$", stripped):
            return ""
        return clean_whitespace(stripped)

    def _next_nonempty_plain_line(
        self, lines: list[str], start_index: int
    ) -> tuple[int | None, str]:
        for idx in range(start_index, len(lines)):
            stripped = lines[idx].strip()
            if not stripped:
                continue
            if self.PAGE_MARKER_PATTERN.match(stripped):
                return None, ""
            return idx, self._strip_heading_marker(stripped)
        return None, ""

    def _is_probably_toc_page(self, lines: list[str], page_marker_index: int) -> bool:
        page_lines: list[str] = []
        for line in lines[page_marker_index + 1 :]:
            stripped = line.strip()
            if self.PAGE_MARKER_PATTERN.match(stripped):
                break
            if stripped:
                page_lines.append(self._strip_heading_marker(stripped))
            if len(page_lines) >= 30:
                break
        if not page_lines:
            return False
        if any(
            line.lower() in {"contents", "table of contents"} for line in page_lines[:5]
        ):
            return True
        trailing_page_number_lines = sum(
            1 for line in page_lines[:25] if re.search(r"\s+\d{1,4}$", line)
        )
        return trailing_page_number_lines >= 10

    def _is_probable_chapter_title_fragment(self, line: str) -> bool:
        title = (line or "").strip()
        if not title:
            return False
        if title.lower() in {"contents", "table of contents"}:
            return False
        if self.SECTION_CODE_PATTERN.match(title) or self.EXERCISES_PATTERN.match(
            title
        ):
            return False
        if re.match(r"^(?:\d+\.)+\d+\s+\w+", title):
            return False
        if re.search(r"\s+\d{1,4}$", title):
            return False
        if len(title.split()) > 12:
            return False
        return True

    def _is_likely_nonstructural_heading(self, heading_body: str) -> bool:
        body = (heading_body or "").strip()
        if not body:
            return False
        first_alpha = re.search(r"[A-Za-z]", body)
        if not first_alpha:
            return False
        if first_alpha.group(0).islower():
            return True
        return False

    def _looks_like_content_after_running_chapter_header(self, line: str) -> bool:
        line = (line or "").strip()
        if not line:
            return False
        if self.SECTION_CODE_PATTERN.match(line) or self.EXERCISES_PATTERN.match(line):
            return True
        if re.match(r"^(?:\d+\.)+\d+\s+\w+", line):
            return True
        return False

    def _normalize_heading_key(self, heading: str) -> str:
        return clean_whitespace(heading).casefold()

    def _is_probable_section_heading(self, section_code: str, title: str) -> bool:
        title = title.strip()
        if not title:
            return False
        if re.search(r"\s+\d{1,4}$", title):
            return False
        if len(title.split()) > 12:
            return False
        if re.search(
            r"\b(definition|notation|lemma|theorem|proposition|corollary|property|example|exercise)\b",
            title,
            flags=re.IGNORECASE,
        ):
            return False
        if section_code[-1] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return False
        return True


class VLMCallError(RuntimeError):
    def __init__(self, message: str, response_text: str = ""):
        super().__init__(message)
        self.response_text = response_text


class PDFToMarkdownOCR:
    HARD_CHECK_FAILURES = {
        "unbalanced_dollar_count",
        "latex_begin_end_mismatch",
        "many_garbled_characters",
        "many_box_symbols",
        "repeated_line_candidates",
        "finish_reason_length",
    }

    TOC_EXTRA_PROMPT = read_text(PROMPT_DIR / "ocr_table_of_contents.md")

    def __init__(self, config: OCRRunConfig, logger: ProjectLogger | None = None):
        self.config = config
        self.app_config = config.app_config or get_config()
        self.logger = logger or ProjectLogger(
            self.app_config.log_dir("ocr"),
            run_name=f"ocr_{config.pdf_path.stem}",
        )
        self.quality_checker = OCRQualityChecker()
        self.structure_normalizer = OCRMarkdownStructureNormalizer()
        self._clients: dict[tuple[str, str, str], Any] = {}

    def run(self) -> dict[str, Any]:
        pdf_path = self.config.pdf_path.resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(str(pdf_path))
        method = self._normalize_method(self.config.method)

        document_id = self.config.document_id or pdf_path.stem
        if Path(document_id).name != document_id or document_id in {".", ".."}:
            raise ValueError("document_id must be a single safe path component.")
        run_dir = ensure_dir(self.config.output_dir / document_id)
        final_md_path = run_dir / f"{document_id}.md"
        report_path = run_dir / f"{document_id}_quality_report.json"
        structure_report_path = run_dir / f"{document_id}_structure_report.json"
        usage_path = run_dir / "usage.json"
        manifest_path = run_dir / "manifest.json"
        source_metadata_path = run_dir / "source_metadata.json"

        page_count = self.get_pdf_page_count(pdf_path)
        first_page, last_page = self._resolve_page_range(page_count)
        print(f"[ocr] input: {pdf_path}")
        print(f"[ocr] pages: {page_count}; processing pages {first_page}-{last_page}")
        print(f"[ocr] method: {method}")
        print(f"[ocr] output: {run_dir}")
        self.logger.info(
            "OCR run started",
            pdf_path=str(pdf_path),
            method=method,
            resume=self.config.resume,
            resume_from_page=self.config.resume_from_page,
            page_count=page_count,
            first_page=first_page,
            last_page=last_page,
            output_dir=str(run_dir),
        )

        image_dir = run_dir / "images"
        image_paths = self.render_pdf_pages(pdf_path, image_dir, first_page, last_page)
        (
            existing_page_results,
            existing_usage_results,
            pending_image_paths,
            resume_report,
        ) = self.prepare_resume_state(
            run_dir=run_dir,
            image_paths=image_paths,
            method=method,
        )
        if resume_report["enabled"]:
            print(
                f"[ocr] resume enabled: loaded {resume_report['loaded_page_count']} existing pages; "
                f"pending {resume_report['pending_page_count']} pages",
                flush=True,
            )
        page_results, usage_results, repair_report = self.run_ocr_mode(
            method, pending_image_paths, run_dir
        )
        all_page_results = self.merge_page_results(
            existing_page_results, page_results, image_paths
        )
        all_usage_results = existing_usage_results + usage_results
        usage = self._build_vlm_usage(
            final_page_results=all_page_results,
            usage_page_results=all_usage_results,
            method=method,
            repair_report=repair_report,
            resume_report=resume_report,
        )

        final_markdown = self.combine_page_results(
            all_page_results, first_page, last_page
        )
        final_markdown, structure_report = self.structure_normalizer.normalize(
            final_markdown
        )
        write_text(final_md_path, final_markdown)
        print(f"[ocr] saved markdown: {final_md_path}")
        write_json(structure_report_path, structure_report)
        print(f"[ocr] saved structure report: {structure_report_path}")

        quality_report = self.quality_checker.check(
            final_markdown, page_count=last_page - first_page + 1
        )
        write_json(report_path, quality_report)
        print(f"[ocr] saved quality report: {report_path}")
        if quality_report["issues"]:
            print(f"[ocr] quality issues: {', '.join(quality_report['issues'])}")
        else:
            print("[ocr] quality issues: none")

        write_json(usage_path, usage)
        print(f"[ocr] saved usage: {usage_path}")

        manifest = {
            "created_or_updated_at": utc_now(),
            "source_pdf": str(pdf_path),
            "page_count": page_count,
            "processed_page_range": [first_page, last_page],
            "method": method,
            "final_markdown": str(final_md_path),
            "quality_report": str(report_path),
            "structure_report": str(structure_report_path),
            "usage": str(usage_path),
            "source_metadata": str(source_metadata_path)
            if source_metadata_path.exists()
            else "",
            "image_dir": str(image_dir),
            "resume_summary": resume_report,
            "quality_summary": {
                "issues": quality_report.get("issues", []),
                "char_count": quality_report.get("char_count"),
                "chars_per_page": quality_report.get("chars_per_page"),
                "garbled_count": quality_report.get("garbled_count"),
                "box_symbol_count": quality_report.get("box_symbol_count"),
            },
            "structure_summary": structure_report,
        }
        write_json(manifest_path, manifest)
        self.logger.info(
            "OCR run finished",
            manifest_path=str(manifest_path),
            markdown_path=str(final_md_path),
        )
        return manifest

    def prepare_resume_state(
        self,
        run_dir: Path,
        image_paths: list[Path],
        method: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path], dict[str, Any]]:
        if not self.config.resume:
            return (
                [],
                [],
                image_paths,
                {
                    "enabled": False,
                    "resume_from_page": None,
                    "loaded_page_count": 0,
                    "pending_page_count": len(image_paths),
                    "loaded_pages": [],
                    "pending_pages": [
                        self._page_number_from_image(path) for path in image_paths
                    ],
                },
            )

        loaded_page_results: list[dict[str, Any]] = []
        loaded_usage_results: list[dict[str, Any]] = []
        pending_image_paths: list[Path] = []
        loaded_pages: list[int] = []
        pending_pages: list[int] = []
        resume_from_page = self.config.resume_from_page

        for image_path in image_paths:
            page_number = self._page_number_from_image(image_path)
            if resume_from_page is not None and page_number >= resume_from_page:
                pending_image_paths.append(image_path)
                pending_pages.append(page_number)
                continue

            loaded = self.load_existing_page_result(
                run_dir=run_dir, page_number=page_number, method=method
            )
            if loaded is None:
                if resume_from_page is not None and page_number < resume_from_page:
                    raise FileNotFoundError(
                        f"Resume requested from page {resume_from_page}, but page {page_number:04d} has no existing OCR result."
                    )
                pending_image_paths.append(image_path)
                pending_pages.append(page_number)
                continue

            page_result, usage_result = loaded
            if page_result.get("status") not in {"ok", "basic_check_failed_kept"}:
                pending_image_paths.append(image_path)
                pending_pages.append(page_number)
                continue
            loaded_page_results.append(page_result)
            loaded_usage_results.append(usage_result)
            loaded_pages.append(page_number)

        return (
            loaded_page_results,
            loaded_usage_results,
            pending_image_paths,
            {
                "enabled": True,
                "resume_from_page": resume_from_page,
                "loaded_page_count": len(loaded_pages),
                "pending_page_count": len(pending_pages),
                "loaded_pages": loaded_pages,
                "pending_pages": pending_pages,
            },
        )

    def load_existing_page_result(
        self,
        run_dir: Path,
        page_number: int,
        method: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        page_name = f"page_{page_number:04d}"
        final_page_path = run_dir / "pages" / f"{page_name}.md"
        if not final_page_path.exists():
            return None

        fallback_pages_dir = run_dir / "premium_fallback_glm46v" / "pages"
        fallback_raw_dir = run_dir / "premium_fallback_glm46v" / "raw"
        fallback_checks_dir = run_dir / "premium_fallback_glm46v" / "page_checks"
        fallback_page_path = fallback_pages_dir / f"{page_name}.md"
        fallback_raw_path = fallback_raw_dir / f"{page_name}.json"
        fallback_check_path = fallback_checks_dir / f"{page_name}.json"

        use_fallback = method == "premium" and fallback_page_path.exists()
        if method == "fast":
            engine = "glm46v"
            stage = "fast"
            default_model = self.config.glm46v_model
            raw_path = run_dir / "raw" / f"{page_name}.json"
            check_path = run_dir / "page_checks" / f"{page_name}.json"
        elif use_fallback:
            engine = "glm46v"
            stage = "premium_fallback"
            default_model = self.config.glm46v_model
            raw_path = fallback_raw_path
            check_path = fallback_check_path
        else:
            engine = "glm5v"
            stage = "premium"
            default_model = self.config.glm5v_model
            raw_path = run_dir / "raw" / f"{page_name}.json"
            check_path = run_dir / "page_checks" / f"{page_name}.json"

        markdown = read_text(final_page_path).strip()
        response_json = read_json(raw_path, default={}) if raw_path.exists() else {}
        check = (
            read_json(check_path, default={})
            if check_path.exists()
            else self.quality_checker.check_page_basic(markdown)
        )
        usage = response_json.get("usage") or {}
        status, error = self.infer_existing_page_status(markdown, check)
        attempts = 1 if status == "ok" else self.config.page_retry_attempts
        model = str(response_json.get("model") or default_model)

        result = {
            "page": page_number,
            "engine": engine,
            "model": model,
            "stage": stage,
            "status": status,
            "attempts": attempts,
            "markdown_path": str(final_page_path),
            "raw_path": str(raw_path) if raw_path.exists() else "",
            "check_path": str(check_path) if check_path.exists() else "",
            "usage": usage,
            "usage_attempts": [usage] if usage else [],
            "markdown": markdown,
            "check": check,
            "error": error,
            "premium_fallback_used": use_fallback,
            "loaded_from_resume": True,
        }
        usage_result = dict(result)
        return result, usage_result

    def infer_existing_page_status(
        self, markdown: str, check: dict[str, Any]
    ) -> tuple[str, str]:
        failure_match = re.search(
            r"\[OCR failed:\s*(.*?)\]", markdown, flags=re.IGNORECASE | re.DOTALL
        )
        if failure_match:
            return "call_failed", failure_match.group(1).strip()
        if check.get("passed_basic_check") is False:
            return "basic_check_failed_kept", "basic_check_failed_after_retries"
        return "ok", ""

    def merge_page_results(
        self,
        existing_page_results: list[dict[str, Any]],
        new_page_results: list[dict[str, Any]],
        image_paths: list[Path],
    ) -> list[dict[str, Any]]:
        merged_by_page: dict[int, dict[str, Any]] = {}
        for result in existing_page_results:
            merged_by_page[int(result["page"])] = result
        for result in new_page_results:
            merged_by_page[int(result["page"])] = result
        return [
            merged_by_page[self._page_number_from_image(path)] for path in image_paths
        ]

    def _normalize_method(self, method: str) -> str:
        aliases = {
            "glm46v": "fast",
            "glm5v": "premium",
            "glm5v-turbo": "premium",
            "glm-5v-turbo": "premium",
        }
        normalized = aliases.get(method, method)
        if normalized not in {"fast", "premium"}:
            raise ValueError("OCR method must be 'fast' or 'premium'.")
        return normalized

    def get_pdf_page_count(self, pdf_path: str | Path) -> int:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)

    def render_pdf_pages(
        self, pdf_path: Path, image_dir: Path, first_page: int, last_page: int
    ) -> list[Path]:
        ensure_dir(image_dir)
        scale = self.config.render_dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        image_paths: list[Path] = []

        with fitz.open(pdf_path) as pdf:
            for page_number in range(first_page, last_page + 1):
                image_path = image_dir / f"page_{page_number:04d}.jpg"
                image_paths.append(image_path)
                if image_path.exists():
                    continue
                page = pdf.load_page(page_number - 1)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                mode = "RGB" if pix.n < 4 else "RGBA"
                image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(
                    image_path,
                    format="JPEG",
                    quality=self.config.jpeg_quality,
                    optimize=True,
                )
                print(
                    f"[render] saved {image_path.name} ({image.width}x{image.height})",
                    flush=True,
                )
        return image_paths

    def run_ocr_mode(
        self, method: str, image_paths: list[Path], run_dir: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        if method == "fast":
            results = self.run_vlm_pages(
                image_paths=image_paths,
                run_dir=run_dir,
                model=self.config.glm46v_model,
                engine="glm46v",
                stage="fast",
                workers=self.config.parallel_workers,
                pages_dir=run_dir / "pages",
                raw_dir=run_dir / "raw",
                checks_dir=run_dir / "page_checks",
                max_tokens=self.config.glm46v_max_tokens,
                retry_delay_seconds=self.config.retry_delay_seconds,
                call_delay_seconds=0.0,
            )
            return (
                results,
                results,
                {"mode": "fast", "fallback_pages": [], "fallback_count": 0},
            )

        premium_results = self.run_vlm_pages(
            image_paths=image_paths,
            run_dir=run_dir,
            model=self.config.glm5v_model,
            engine="glm5v",
            stage="premium",
            workers=1,
            pages_dir=run_dir / "pages",
            raw_dir=run_dir / "raw",
            checks_dir=run_dir / "page_checks",
            max_tokens=self.config.glm5v_max_tokens,
            retry_delay_seconds=self.config.glm5v_retry_delay_seconds,
            call_delay_seconds=self.config.glm5v_call_delay_seconds,
        )
        fallback_pages = [
            result["page"]
            for result in premium_results
            if result.get("status") == "call_failed"
        ]
        fallback_paths = [
            path
            for path in image_paths
            if self._page_number_from_image(path) in fallback_pages
        ]
        fallback_results: list[dict[str, Any]] = []
        if fallback_paths:
            print(
                f"[premium] falling back to {self.config.glm46v_model} "
                f"for {len(fallback_paths)} API-failed pages"
            )
            fallback_results = self.run_vlm_pages(
                image_paths=fallback_paths,
                run_dir=run_dir,
                model=self.config.glm46v_model,
                engine="glm46v",
                stage="premium_fallback",
                workers=self.config.parallel_workers,
                pages_dir=run_dir / "premium_fallback_glm46v" / "pages",
                raw_dir=run_dir / "premium_fallback_glm46v" / "raw",
                checks_dir=run_dir / "premium_fallback_glm46v" / "page_checks",
                max_tokens=self.config.glm46v_max_tokens,
                retry_delay_seconds=self.config.retry_delay_seconds,
                call_delay_seconds=0.0,
            )
        final_results = self.merge_premium_results(
            premium_results, fallback_results, run_dir
        )
        repair_report = {
            "mode": "premium",
            "fallback_pages": fallback_pages,
            "fallback_count": len(fallback_pages),
        }
        return final_results, premium_results + fallback_results, repair_report

    def run_vlm_pages(
        self,
        image_paths: list[Path],
        run_dir: Path,
        model: str,
        engine: str,
        stage: str,
        workers: int,
        pages_dir: Path,
        raw_dir: Path,
        checks_dir: Path,
        max_tokens: int,
        retry_delay_seconds: float,
        call_delay_seconds: float,
    ) -> list[dict[str, Any]]:
        ensure_dir(pages_dir)
        ensure_dir(raw_dir)
        ensure_dir(checks_dir)
        prompt = read_text(self.config.ocr_prompt_path).strip()
        results_by_page: dict[int, dict[str, Any]] = {}

        print(
            f"[{stage}] processing {len(image_paths)} pages with {workers} workers using {model}"
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_image = {
                executor.submit(
                    self.process_vlm_page,
                    image_path,
                    prompt,
                    pages_dir,
                    raw_dir,
                    checks_dir,
                    model,
                    engine,
                    stage,
                    max_tokens,
                    retry_delay_seconds,
                    call_delay_seconds,
                ): image_path
                for image_path in image_paths
            }
            completed_count = 0
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                page_number = self._page_number_from_image(image_path)
                try:
                    result = future.result()
                except Exception as exc:
                    result = self.failed_page_result(
                        image_path=image_path,
                        pages_dir=pages_dir,
                        checks_dir=checks_dir,
                        engine=engine,
                        stage=stage,
                        reason=f"worker_crash: {exc}",
                    )
                results_by_page[page_number] = result
                completed_count += 1
                print(
                    f"[{stage}] progress {completed_count}/{len(image_paths)}: "
                    f"page {page_number:04d} -> {result['engine']} ({result['status']})",
                    flush=True,
                )

        return [
            results_by_page[self._page_number_from_image(path)] for path in image_paths
        ]

    def process_vlm_page(
        self,
        image_path: Path,
        prompt: str,
        pages_dir: Path,
        raw_dir: Path,
        checks_dir: Path,
        model: str,
        engine: str,
        stage: str,
        max_tokens: int,
        retry_delay_seconds: float,
        call_delay_seconds: float,
    ) -> dict[str, Any]:
        page_number = self._page_number_from_image(image_path)
        page_md_path = pages_dir / f"page_{page_number:04d}.md"
        raw_path = raw_dir / f"page_{page_number:04d}.json"
        check_path = checks_dir / f"page_{page_number:04d}.json"

        last_call_error = ""
        last_markdown = ""
        last_response: dict[str, Any] | None = None
        last_check: dict[str, Any] | None = None
        usage_attempts: list[dict[str, Any]] = []
        for attempt in range(1, self.config.page_retry_attempts + 1):
            try:
                if call_delay_seconds > 0:
                    time.sleep(call_delay_seconds)
                response = self.call_vlm(
                    image_path, prompt, page_number, model, max_tokens
                )
                usage_attempts.append(response.get("usage") or {})
                markdown = self.extract_message_text(response)
                if not markdown.strip():
                    raise VLMCallError(f"page {page_number:04d} returned empty content")
                if not markdown.lstrip().startswith("<!-- page"):
                    markdown = f"<!-- page {page_number:04d} -->\n" + markdown.strip()
                markdown = self.normalize_image_placeholders(markdown)
                check = self.quality_checker.check_page_basic(markdown)
                finish_reason = self._response_finish_reason(response)
                if finish_reason == "length":
                    check["critical_issues"].append("finish_reason_length")
                    check["issues"].append("finish_reason_length")
                    check["passed_basic_check"] = False
                last_markdown = markdown
                last_response = response
                last_check = check
                if check["passed_basic_check"]:
                    write_text(page_md_path, markdown.strip() + "\n")
                    write_text(
                        raw_path, json.dumps(response, ensure_ascii=False, indent=2)
                    )
                    write_json(check_path, check)
                    return {
                        "page": page_number,
                        "engine": engine,
                        "model": model,
                        "stage": stage,
                        "status": "ok",
                        "attempts": attempt,
                        "markdown_path": str(page_md_path),
                        "raw_path": str(raw_path),
                        "check_path": str(check_path),
                        "usage": response.get("usage") or {},
                        "usage_attempts": usage_attempts,
                        "markdown": markdown.strip(),
                        "check": check,
                    }
                print(
                    f"[{stage}] page {page_number:04d} kept after basic check warning: "
                    f"{check['critical_issues']}",
                    flush=True,
                )
                write_text(page_md_path, markdown.strip() + "\n")
                write_text(raw_path, json.dumps(response, ensure_ascii=False, indent=2))
                write_json(check_path, check)
                return {
                    "page": page_number,
                    "engine": engine,
                    "model": model,
                    "stage": stage,
                    "status": "basic_check_failed_kept",
                    "attempts": attempt,
                    "markdown_path": str(page_md_path),
                    "raw_path": str(raw_path),
                    "check_path": str(check_path),
                    "usage": response.get("usage") or {},
                    "usage_attempts": usage_attempts,
                    "markdown": markdown.strip(),
                    "check": check,
                    "error": "basic_check_failed_kept_without_retry",
                }
            except Exception as exc:
                last_call_error = str(exc)
                print(
                    f"[{stage}] page {page_number:04d} attempt {attempt} call failed: {exc}",
                    flush=True,
                )
            if attempt < self.config.page_retry_attempts:
                time.sleep(retry_delay_seconds)

        if last_markdown and last_check is not None and last_response is not None:
            write_text(page_md_path, last_markdown.strip() + "\n")
            write_text(
                raw_path, json.dumps(last_response, ensure_ascii=False, indent=2)
            )
            write_json(check_path, last_check)
            return {
                "page": page_number,
                "engine": engine,
                "model": model,
                "stage": stage,
                "status": "basic_check_failed_kept",
                "attempts": self.config.page_retry_attempts,
                "markdown_path": str(page_md_path),
                "raw_path": str(raw_path),
                "check_path": str(check_path),
                "usage": last_response.get("usage") or {},
                "usage_attempts": usage_attempts,
                "markdown": last_markdown.strip(),
                "check": last_check,
                "error": "basic_check_failed_after_retries",
            }

        return self.failed_page_result(
            image_path=image_path,
            pages_dir=pages_dir,
            checks_dir=checks_dir,
            engine=engine,
            stage=stage,
            reason=last_call_error or "empty_response_after_retries",
            usage_attempts=usage_attempts,
        )

    def failed_page_result(
        self,
        image_path: Path,
        pages_dir: Path,
        checks_dir: Path,
        engine: str,
        stage: str,
        reason: str,
        usage_attempts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        page_number = self._page_number_from_image(image_path)
        page_md_path = pages_dir / f"page_{page_number:04d}.md"
        check_path = checks_dir / f"page_{page_number:04d}.json"
        markdown = f"<!-- page {page_number:04d} -->\n[OCR failed: {reason}]"
        check = self.quality_checker.check_page_basic(markdown)
        write_text(page_md_path, markdown.strip() + "\n")
        write_json(check_path, check)
        return {
            "page": page_number,
            "engine": engine,
            "model": "",
            "stage": stage,
            "status": "call_failed",
            "attempts": self.config.page_retry_attempts,
            "markdown_path": str(page_md_path),
            "raw_path": "",
            "check_path": str(check_path),
            "usage": {},
            "usage_attempts": usage_attempts or [],
            "markdown": markdown,
            "check": check,
            "error": reason,
        }

    def call_vlm(
        self,
        image_path: Path,
        prompt: str,
        page_number: int,
        model: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        client = self.get_vlm_client(model)
        page_prompt = prompt
        if page_number in self.config.toc_pages:
            page_prompt = prompt + self.TOC_EXTRA_PROMPT
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + self.encode_base64(image_path)
                },
            },
            {
                "type": "text",
                "text": (
                    page_prompt
                    + f"\n\nThe input is page {page_number} from a mathematical document. "
                    f"Output this page only. Start with `<!-- page {page_number:04d} -->`."
                ),
            },
        ]
        extra = {"thinking": {"type": "disabled"}} if model.lower().startswith("glm") else {}
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=max_tokens,
            extra_body=extra,
        )
        return self.response_to_dict(response)

    def get_vlm_client(self, model: str) -> Any:
        provider_name = (
            self.config.fast_provider
            if model == self.config.glm46v_model
            else self.config.premium_provider
        )
        provider = self.app_config.provider(provider_name, require_api_key=True)
        api_key = self.config.provider_api_key or provider.api_key
        base_url = (
            self.config.provider_base_url
            if self.config.provider_base_url is not None
            else provider.base_url
        )
        cache_key = (provider_name, api_key, base_url)
        client = self._clients.get(cache_key)
        if client is not None:
            return client

        try:
            from openai import OpenAI
            import httpx
        except ImportError as exc:
            raise ImportError(
                "OCR requires the openai and httpx packages."
            ) from exc
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.config.glm46v_timeout_seconds,
            "max_retries": 0,
            "http_client": httpx.Client(trust_env=provider.trust_env),
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        self._clients[cache_key] = client
        return client

    def merge_premium_results(
        self,
        premium_results: list[dict[str, Any]],
        fallback_results: list[dict[str, Any]],
        run_dir: Path,
    ) -> list[dict[str, Any]]:
        final_pages_dir = ensure_dir(run_dir / "pages")
        fallback_by_page = {int(result["page"]): result for result in fallback_results}
        final_results = []
        for premium_result in premium_results:
            page = int(premium_result["page"])
            fallback_result = fallback_by_page.get(page)
            if fallback_result and fallback_result.get("status") in {
                "ok",
                "basic_check_failed_kept",
            }:
                final_result = dict(fallback_result)
                final_result["premium_fallback_used"] = True
                final_page_path = final_pages_dir / f"page_{page:04d}.md"
                write_text(
                    final_page_path, final_result.get("markdown", "").strip() + "\n"
                )
                final_result["markdown_path"] = str(final_page_path)
            else:
                final_result = dict(premium_result)
                final_result["premium_fallback_used"] = False
            final_results.append(final_result)
        return final_results

    def combine_page_results(
        self, page_results: list[dict[str, Any]], first_page: int, last_page: int
    ) -> str:
        by_page = {int(result["page"]): result for result in page_results}
        parts = []
        for page_number in range(first_page, last_page + 1):
            result = by_page.get(page_number)
            if result is None:
                parts.append(f"<!-- page {page_number:04d} -->\n[missing OCR page]")
                continue
            markdown = (result.get("markdown") or "").strip()
            if not markdown.lstrip().startswith("<!-- page"):
                markdown = f"<!-- page {page_number:04d} -->\n" + markdown
            markdown = self.normalize_image_placeholders(markdown)
            parts.append(markdown)
        return "\n\n".join(parts).strip() + "\n"

    def normalize_image_placeholders(self, markdown: str) -> str:
        image_pattern = r"!\[[^\]\n]*\]\([^)]+\)"
        text = re.sub(rf"({image_pattern})", r"\n\n\1\n\n", markdown)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _build_vlm_usage(
        self,
        final_page_results: list[dict[str, Any]],
        usage_page_results: list[dict[str, Any]],
        method: str,
        repair_report: dict[str, Any],
        resume_report: dict[str, Any],
    ) -> dict[str, Any]:
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage_events = []
        for result in usage_page_results:
            usage = result.get("usage") or {}
            usage_attempts = result.get("usage_attempts") or [usage]
            for attempt_usage in usage_attempts:
                for key in total:
                    total[key] += int((attempt_usage or {}).get(key) or 0)
            usage_events.append(
                {
                    "page": result.get("page"),
                    "engine": result.get("engine"),
                    "model": result.get("model"),
                    "stage": result.get("stage"),
                    "status": result.get("status"),
                    "attempts": result.get("attempts"),
                    "usage": usage,
                    "usage_attempts": usage_attempts,
                    "critical_issues": (result.get("check") or {}).get(
                        "critical_issues", []
                    ),
                    "error": result.get("error", ""),
                }
            )

        final_pages = []
        for result in final_page_results:
            final_pages.append(
                {
                    "page": result.get("page"),
                    "engine": result.get("engine"),
                    "model": result.get("model"),
                    "stage": result.get("stage"),
                    "status": result.get("status"),
                    "attempts": result.get("attempts"),
                    "usage": result.get("usage") or {},
                    "critical_issues": (result.get("check") or {}).get(
                        "critical_issues", []
                    ),
                    "premium_fallback_used": result.get("premium_fallback_used", False),
                }
            )

        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "method": method,
            "models": {
                "fast": self.config.glm46v_model,
                "premium": self.config.glm5v_model,
            },
            "parallel_workers": {
                "glm46v": self.config.parallel_workers,
                "glm5v": 1,
            },
            "total": total,
            "pages": final_pages,
            "usage_events": usage_events,
            "repair_report": repair_report,
            "resume_report": resume_report,
        }

    def _resolve_page_range(self, page_count: int) -> tuple[int, int]:
        first_page = max(1, self.config.first_page)
        last_page = self.config.last_page or page_count
        last_page = min(page_count, last_page)
        if first_page > last_page:
            raise ValueError(
                f"Invalid page range: {first_page}-{last_page} for {page_count} pages."
            )
        return first_page, last_page

    def response_to_dict(self, response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "dict"):
            return response.dict()
        if hasattr(response, "json"):
            return json.loads(response.json())
        return json.loads(json.dumps(response, default=str))

    def extract_message_text(self, response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            pieces: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    pieces.append(item["text"])
            return "\n".join(pieces).strip()
        return str(content).strip()

    def _response_finish_reason(self, response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("finish_reason") or "")

    def encode_base64(self, path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _page_number_from_image(self, image_path: Path) -> int:
        return self._page_number_from_text(image_path.stem)

    def _page_number_from_text(self, text: str) -> int:
        match = re.search(r"(\d+)$", text)
        if not match:
            return 0
        return int(match.group(1))


def main() -> None:
    from scripts.pipeline.ocr_extraction import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()

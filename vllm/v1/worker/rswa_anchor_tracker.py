# SPDX-License-Identifier: Apache-2.0
"""Decode-side DocLang grammar-anchor tracking for R-SWA.

implementation) so the vLLM worker can reproduce the anchor liveness the model
was *trained* with: decode tokens attend to the prompt/image reference, a
sliding window of recent output tokens, and a small set of structurally *live*
DocLang anchor tokens (open-tag stack + location tokens, table header band,
previous-row structural cells, thread/xref openers, last page break, heading
trail).

The engine is driven per scheduler step by the GPU model runner: for every
running R-SWA request it consumes newly sampled token ids through an
incremental automaton and maintains a per-request byte mask over absolute KV
positions (1 = live anchor, visible past the window). The attention kernel
ORs this mask into the R-SWA visibility predicate.

Config knobs (read from the HF config by the runner / attention backend):

* ``rswa_decode_anchors``   – "none" (legacy prefix+window decode) or
                               "grammar" (run this automaton as trained).
* ``rswa_keep_all_locs``    – every ``<location .../>`` token stays a
                               permanent anchor (global layout map).
* ``rswa_closed_trail_k``   – keep tag+location anchors of the last K *closed*
                               elements alive (0 disables).

Pure-python, CPU-side; a decode step advances each request by one token, so
per-step cost is O(1) automaton work + a few byte writes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_PAIRED_TAGS: dict[str, str] = {
    "<doclang>": "</doclang>",
    "<table>": "</table>",
    "<tabular>": "</tabular>",
    "<list>": "</list>",
    '<list class="ordered">': "</list>",
    "<group>": "</group>",
    "<field_region>": "</field_region>",
    "<field_item>": "</field_item>",
    "<key>": "</key>",
    "<value>": "</value>",
    "<hint>": "</hint>",
    '<picture><label value="': "</picture>",
    '<picture class="chart"><label value="': "</picture>",
    "<text>": "</text>",
    "<caption>": "</caption>",
    "<footnote>": "</footnote>",
    "<formula>": "</formula>",
    '<code><label value="': "</code>",
    "<page_header>": "</page_header>",
    "<page_footer>": "</page_footer>",
    "<content>": "</content>",
    "<custom>": "</custom>",
    "<handwriting>": "</handwriting>",
}

_TABLE_OPEN_STRINGS = ("<table>", "<tabular>")
_ROW_STRUCT_STRINGS = (
    "<fcel/>", "<ecel/>", "<ched/>", "<rhed/>", "<corn/>",
    "<srow/>", "<lcel/>", "<ucel/>", "<xcel/>", "<nl/>",
)
_NL_STRING = "<nl/>"
_THREAD_PREFIX_STRINGS = ('<thread thread_id="', '<xref thread_id="')
_PAGE_BREAK_STRINGS = ("<page_break>", "<page_break/>")
_HEADING_LEVEL_STRINGS = {f'<heading level="{k}">': k for k in range(1, 7)}
_HEADING_CLOSE_STRING = "</heading>"

_MAX_HEADER_ANCHORS = 512
_MAX_ROW_ANCHORS = 512
_MAX_THREAD_ATTR_TOKENS = 8


def _tid(tokenizer, tok: str) -> int | None:
    tid = tokenizer.convert_tokens_to_ids(tok)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or tid == unk:
        return None
    return int(tid)


@dataclass(frozen=True)
class DocLangTagIds:
    open_to_close: dict[int, int]
    close_ids: frozenset[int]
    table_open_ids: frozenset[int]
    row_struct_ids: frozenset[int]
    nl_id: int | None
    thread_prefix_ids: frozenset[int]
    attr_close_ids: frozenset[int]
    page_break_ids: frozenset[int]
    heading_open_levels: dict[int, int]
    heading_close_ids: frozenset[int]
    location_ids: frozenset[int]

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "DocLangTagIds":
        open_to_close: dict[int, int] = {}
        for open_s, close_s in _PAIRED_TAGS.items():
            o, c = _tid(tokenizer, open_s), _tid(tokenizer, close_s)
            if o is not None and c is not None:
                open_to_close[o] = c
        table_open_ids = frozenset(
            t for s in _TABLE_OPEN_STRINGS if (t := _tid(tokenizer, s)) is not None
        )
        row_struct_ids = frozenset(
            t for s in _ROW_STRUCT_STRINGS if (t := _tid(tokenizer, s)) is not None
        )
        thread_prefix_ids = frozenset(
            t for s in _THREAD_PREFIX_STRINGS if (t := _tid(tokenizer, s)) is not None
        )
        page_break_ids = frozenset(
            t for s in _PAGE_BREAK_STRINGS if (t := _tid(tokenizer, s)) is not None
        )
        heading_open_levels = {
            t: lvl
            for s, lvl in _HEADING_LEVEL_STRINGS.items()
            if (t := _tid(tokenizer, s)) is not None
        }
        heading_close_ids = frozenset(
            t for s in (_HEADING_CLOSE_STRING,) if (t := _tid(tokenizer, s)) is not None
        )
        location_ids: set[int] = set()
        attr_close_ids: set[int] = set()
        for tok, idx in tokenizer.get_vocab().items():
            if tok.startswith('<location value="') and tok.endswith('"/>'):
                location_ids.add(int(idx))
            elif tok in ('">', '"/>'):
                attr_close_ids.add(int(idx))
        return cls(
            open_to_close=open_to_close,
            close_ids=frozenset(open_to_close.values()),
            table_open_ids=table_open_ids,
            row_struct_ids=row_struct_ids,
            nl_id=_tid(tokenizer, _NL_STRING),
            thread_prefix_ids=thread_prefix_ids,
            attr_close_ids=frozenset(attr_close_ids),
            page_break_ids=page_break_ids,
            heading_open_levels=heading_open_levels,
            heading_close_ids=heading_close_ids,
            location_ids=frozenset(location_ids),
        )


@dataclass
class _Frame:
    open_id: int
    close_id: int
    positions: list[int]
    collecting_locations: bool = True
    is_table: bool = False
    in_header: bool = False
    header_positions: list[int] = field(default_factory=list)
    prev_row: list[int] = field(default_factory=list)
    cur_row: list[int] = field(default_factory=list)


class DocLangAnchorTracker:
    """Incremental DocLang automaton emitting anchor birth/death events.

    Variant knobs on top of the training-side reference automaton:

    * ``keep_all_locs`` – every location token becomes a permanent anchor
      (excluded from all deaths).
    * ``closed_trail_k`` – a closed element's tag+location anchors go into a
      FIFO trail of size K instead of dying immediately; they die when K
      newer elements have closed. Table-internal anchors (header band, rows)
      still die with the table.
    """

    def __init__(
        self,
        tag_ids: DocLangTagIds,
        start_pos: int = 0,
        *,
        keep_all_locs: bool = False,
        closed_trail_k: int = 0,
    ):
        self.tags = tag_ids
        self.pos = start_pos
        self.stack: list[_Frame] = []
        self.heading_trail: list[tuple[int, list[int]]] = []
        self.open_heading: list[int] | None = None
        self.open_heading_level: int = 0
        self.last_page_break: int | None = None
        self.thread_attr_remaining: int = 0
        self.keep_all_locs = keep_all_locs
        self.closed_trail_k = int(closed_trail_k)
        self._permanent: set[int] = set()
        self._closed_trail: deque[list[int]] = deque()

    def _innermost_table(self) -> _Frame | None:
        for frame in reversed(self.stack):
            if frame.is_table:
                return frame
        return None

    def _pop_frame(self, frame: _Frame, deaths: list[int]) -> None:
        if self.closed_trail_k > 0:
            # Element tag+loc anchors graduate into the closed-elements trail.
            self._closed_trail.append(frame.positions)
            while len(self._closed_trail) > self.closed_trail_k:
                deaths.extend(self._closed_trail.popleft())
        else:
            deaths.extend(frame.positions)
        if frame.is_table:
            deaths.extend(frame.header_positions)
            deaths.extend(frame.prev_row)
            deaths.extend(frame.cur_row)

    def step(self, token_id: int) -> tuple[list[int], list[int]]:
        tags = self.tags
        i = self.pos
        births: list[int] = []
        deaths: list[int] = []

        if (
            token_id in tags.location_ids
            and self.stack
            and self.stack[-1].collecting_locations
        ):
            self.stack[-1].positions.append(i)
            births.append(i)
        elif self.stack and token_id not in tags.location_ids:
            self.stack[-1].collecting_locations = False

        # keep_all_locs: any location token anywhere is a permanent anchor.
        if self.keep_all_locs and token_id in tags.location_ids:
            births.append(i)
            self._permanent.add(i)

        if token_id in tags.thread_prefix_ids:
            births.append(i)
            self.thread_attr_remaining = _MAX_THREAD_ATTR_TOKENS
        elif self.thread_attr_remaining > 0:
            births.append(i)
            if token_id in tags.attr_close_ids:
                self.thread_attr_remaining = 0
            else:
                self.thread_attr_remaining -= 1

        if token_id in tags.page_break_ids:
            if self.last_page_break is not None:
                deaths.append(self.last_page_break)
            self.last_page_break = i
            births.append(i)

        if token_id in tags.heading_open_levels:
            level = tags.heading_open_levels[token_id]
            kept: list[tuple[int, list[int]]] = []
            for lvl, positions in self.heading_trail:
                if lvl >= level:
                    deaths.extend(positions)
                else:
                    kept.append((lvl, positions))
            self.heading_trail = kept
            self.open_heading = [i]
            self.open_heading_level = level
            births.append(i)
        elif self.open_heading is not None:
            if token_id in tags.heading_close_ids:
                self.open_heading.append(i)
                births.append(i)
                self.heading_trail.append((self.open_heading_level, self.open_heading))
                self.open_heading = None
            elif len(self.open_heading) < _MAX_HEADER_ANCHORS:
                self.open_heading.append(i)
                births.append(i)

        table = self._innermost_table()
        if table is not None and token_id not in tags.table_open_ids:
            if table.in_header:
                if len(table.header_positions) < _MAX_HEADER_ANCHORS:
                    table.header_positions.append(i)
                    births.append(i)
                if token_id == tags.nl_id:
                    table.in_header = False
            elif token_id == tags.nl_id:
                deaths.extend(table.prev_row)
                table.prev_row = table.cur_row
                table.prev_row.append(i)
                births.append(i)
                table.cur_row = []
            elif token_id in tags.row_struct_ids:
                if len(table.cur_row) < _MAX_ROW_ANCHORS:
                    table.cur_row.append(i)
                    births.append(i)

        if token_id in tags.open_to_close:
            frame = _Frame(
                open_id=token_id,
                close_id=tags.open_to_close[token_id],
                positions=[i],
                is_table=token_id in tags.table_open_ids,
                in_header=token_id in tags.table_open_ids,
            )
            self.stack.append(frame)
            births.append(i)
        elif token_id in tags.close_ids:
            match_idx = None
            for idx in range(len(self.stack) - 1, -1, -1):
                if self.stack[idx].close_id == token_id:
                    match_idx = idx
                    break
            if match_idx is not None:
                while len(self.stack) > match_idx:
                    self._pop_frame(self.stack.pop(), deaths)

        if self._permanent and deaths:
            deaths = [d for d in deaths if d not in self._permanent]

        self.pos = i + 1
        return births, deaths


class RswaAnchorEngine:
    """Per-request anchor state for the model runner.

    Owns the DocLang tag tables (tokenizer loaded once, lazily) and a tracker
    per running request. ``advance`` replays newly appended output tokens and
    applies births/deaths directly onto the request's byte-mask row (a numpy
    view of the runner's pinned host buffer).
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int,
        *,
        keep_all_locs: bool = False,
        closed_trail_k: int = 0,
    ) -> None:
        self._model_path = model_path
        self._max_model_len = int(max_model_len)
        self._keep_all_locs = keep_all_locs
        self._closed_trail_k = closed_trail_k
        self._tag_ids: DocLangTagIds | None = None
        # req_id -> [tracker, num_output_tokens_processed, mask_row].
        # The engine owns each request's mask row (batch slots reshuffle
        # between steps, so incremental per-slot rows would corrupt).
        self._states: dict[str, list] = {}

    def _get_tag_ids(self) -> DocLangTagIds:
        if self._tag_ids is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, trust_remote_code=True
            )
            self._tag_ids = DocLangTagIds.from_tokenizer(tokenizer)
        return self._tag_ids

    def advance(
        self,
        req_id: str,
        prompt_len: int,
        output_token_ids: list[int],
    ) -> np.ndarray:
        """Consume newly appended output tokens; return the request's
        up-to-date anchor byte mask row (length ``max_model_len``)."""
        state = self._states.get(req_id)
        if state is not None and state[1] > len(output_token_ids):
            self.reset_count = getattr(self, "reset_count", 0) + 1
        if state is None or state[1] > len(output_token_ids):
            # New request, or recomputed from scratch (processed count exceeds
            # current output length): reset and replay.
            state = [
                DocLangAnchorTracker(
                    self._get_tag_ids(),
                    start_pos=prompt_len,
                    keep_all_locs=self._keep_all_locs,
                    closed_trail_k=self._closed_trail_k,
                ),
                0,
                np.zeros(self._max_model_len, dtype=np.uint8),
            ]
            self._states[req_id] = state
        tracker, processed, mask_row = state[0], state[1], state[2]
        n = len(output_token_ids)
        limit = mask_row.shape[0]
        dirty: list = state[3] if len(state) > 3 else None
        for tok in output_token_ids[processed:n]:
            births, deaths = tracker.step(int(tok))
            for p in births:
                if p < limit:
                    mask_row[p] = 1
                    if dirty is not None:
                        dirty.append((p, 1))
            for p in deaths:
                if p < limit:
                    mask_row[p] = 0
                    if dirty is not None:
                        dirty.append((p, 0))
        state[1] = n
        return mask_row

    def advance_delta(
        self,
        req_id: str,
        prompt_len: int,
        output_token_ids: list[int],
    ):
        """Like ``advance`` but returns ``(mask_row, dirty, fresh)``.

        ``dirty`` is the (position, value) changes accumulated since the last
        ``advance_delta`` call for this request; ``fresh`` is True when the
        state was (re)created — the caller must then re-stage the full row.
        Enables incremental GPU-mask updates instead of per-step full-row
        staging (the dominant serving overhead at high concurrency).
        """
        state = self._states.get(req_id)
        fresh = state is None or state[1] > len(output_token_ids)
        if fresh and state is not None:
            self._states.pop(req_id, None)
        if fresh:
            # advance() will recreate; attach the dirty list afterwards.
            row = self.advance(req_id, prompt_len, output_token_ids)
            self._states[req_id].append([])
            return row, [], True
        if len(state) == 3:
            state.append([])
        row = self.advance(req_id, prompt_len, output_token_ids)
        dirty = state[3]
        state[3] = []
        return row, dirty, False

    def prune(self, live_req_ids) -> None:
        """Drop trackers of requests no longer running."""
        stale = [r for r in self._states if r not in live_req_ids]
        for r in stale:
            self._states.pop(r, None)

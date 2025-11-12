from typing import List, Tuple, Optional, Iterable
import re
from transformers import PreTrainedTokenizerBase

def find_all_substrings(haystack: str, needle: str) -> Iterable[Tuple[int, int]]:
    if not needle:
        return
    for m in re.finditer(re.escape(needle), haystack):
        s = m.start()
        e = m.end()
        yield (s, e)

def token_spans_for_char_span(
    offsets: List[Tuple[int, int]],
    start_char: int,
    end_char: int
) -> Optional[Tuple[int, int]]:
    start_tok = None
    end_tok = None
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        if e <= start_char:
            continue
        if s > end_char:
            break
        if start_tok is None:
            start_tok = i
        end_tok = i
    if start_tok is None:
        return None
    return (start_tok, end_tok)

def find_token_spans(
    A: str,
    B: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    ignore_case: bool = False
) -> List[Tuple[int, int]]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Fast tokenizer is required")
        

    if not B:
        return []
    
    B = B.strip()

    assert B in A, f"B not found in A."

    hay = A
    ned = B
    flags = re.IGNORECASE if ignore_case else 0
    matches = list(re.finditer(re.escape(ned), hay, flags=flags))
    if not matches:
        return []

    enc = tokenizer(
        A,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False
    )
    offsets: List[Tuple[int, int]] = enc["offset_mapping"]

    spans: List[Tuple[int, int]] = []
    for m in matches:
        start_char = m.start()
        end_char = m.end()
        tok_span = token_spans_for_char_span(offsets, start_char, end_char)
        if tok_span is not None:
            spans.append(tok_span)

    return spans

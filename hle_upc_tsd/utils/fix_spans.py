"""Automatic best effort span cleaner for SemEval 2021 Toxic Spans."""

import itertools
import string

SPECIAL_CHARACTERS = string.whitespace


def _contiguous_ranges(span_list):
    """
    Extracts continguous runs [1, 2, 3, 5, 6, 7] -> [(1,3), (5,7)].
    Returns begin and end inclusive
    """
    output = []
    for _, span in itertools.groupby(enumerate(span_list), lambda p: p[1] - p[0]):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    return output


def my_fix_spans(spans, text, special_characters=SPECIAL_CHARACTERS, collapse=False):
    """
    Applies minor edits to trim spans and remove singletons.
    If spans begin/end in the middle of a word, correct according to collapse strategy:
        If false, expand spans until word limits; if true collapse until word limits
    """
    cleaned_spans = []
    for begin, end in _contiguous_ranges(spans):
        # Trim spans
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        # Assert word limits
        while 0 < begin < end and text[begin - 1].isalnum():
            offset_move = 1 if collapse else -1
            begin += offset_move
        while len(text) - 1 > end > begin and text[end + 1].isalnum():
            offset_move = -1 if collapse else 1
            end += offset_move
        # Remove singletons
        if end - begin > 1:
            cleaned_spans.extend(range(begin, end + 1))
    return cleaned_spans

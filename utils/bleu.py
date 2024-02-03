# Copyright 2020 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BLEU metric. """

import collections
import math

import re
from functools import lru_cache


class BaseTokenizer:
    """A base dummy tokenizer to derive from."""

    def signature(self):
        """
        Returns a signature for the tokenizer.
        :return: signature string
        """
        return "none"

    def __call__(self, line):
        """
        Tokenizes an input line with the tokenizer.
        :param line: a segment to tokenize
        :return: the tokenized line
        """
        return line


class TokenizerRegexp(BaseTokenizer):
    def signature(self):
        return "re"

    def __init__(self):
        self._re = [
            # language-dependent part (assuming Western languages)
            (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
            # tokenize period and comma unless preceded by a digit
            (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
            # tokenize period and comma unless followed by a digit
            (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
            # tokenize dash when preceded by a digit
            (re.compile(r"([0-9])(-)"), r"\1 \2 "),
            # one space only between words
            # NOTE: Doing this in Python (below) is faster
            # (re.compile(r'\s+'), r' '),
        ]

    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        """Common post-processing tokenizer for `13a` and `zh` tokenizers.
        :param line: a segment to tokenize
        :return: the tokenized line
        """
        for _re, repl in self._re:
            line = _re.sub(repl, line)

        # no leading or trailing spaces, single space within words
        # return ' '.join(line.split())
        # This line is changed with regards to the original tokenizer (seen above) to return individual words
        return line.split()


class Tokenizer13a(BaseTokenizer):
    def signature(self):
        return "13a"

    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        """Tokenizes an input line using a relatively minimal tokenization
        that is however equivalent to mteval-v13a, used by WMT.

        :param line: a segment to tokenize
        :return: the tokenized line
        """

        # language-independent part:
        line = line.replace("<skipped>", "")
        line = line.replace("-\n", "")
        line = line.replace("\n", " ")

        if "&" in line:
            line = line.replace("&quot;", '"')
            line = line.replace("&amp;", "&")
            line = line.replace("&lt;", "<")
            line = line.replace("&gt;", ">")

        return self._post_tokenizer(f" {line} ")


from functools import lru_cache


_UCODE_RANGES = [
    ("\u3400", "\u4db5"),  # CJK Unified Ideographs Extension A, release 3.0
    ("\u4e00", "\u9fa5"),  # CJK Unified Ideographs, release 1.1
    ("\u9fa6", "\u9fbb"),  # CJK Unified Ideographs, release 4.1
    ("\uf900", "\ufa2d"),  # CJK Compatibility Ideographs, release 1.1
    ("\ufa30", "\ufa6a"),  # CJK Compatibility Ideographs, release 3.2
    ("\ufa70", "\ufad9"),  # CJK Compatibility Ideographs, release 4.1
    (
        "\u20000",
        "\u2a6d6",
    ),  # (UTF16) CJK Unified Ideographs Extension B, release 3.1
    (
        "\u2f800",
        "\u2fa1d",
    ),  # (UTF16) CJK Compatibility Supplement, release 3.1
    (
        "\uff00",
        "\uffef",
    ),  # Full width ASCII, full width of English punctuation,
    # half width Katakana, half wide half width kana, Korean alphabet
    ("\u2e80", "\u2eff"),  # CJK Radicals Supplement
    ("\u3000", "\u303f"),  # CJK punctuation mark
    ("\u31c0", "\u31ef"),  # CJK stroke
    ("\u2f00", "\u2fdf"),  # Kangxi Radicals
    ("\u2ff0", "\u2fff"),  # Chinese character structure
    ("\u3100", "\u312f"),  # Phonetic symbols
    ("\u31a0", "\u31bf"),  # Phonetic symbols (Taiwanese and Hakka expansion)
    ("\ufe10", "\ufe1f"),
    ("\ufe30", "\ufe4f"),
    ("\u2600", "\u26ff"),
    ("\u2700", "\u27bf"),
    ("\u3200", "\u32ff"),
    ("\u3300", "\u33ff"),
]


class TokenizerZh(BaseTokenizer):
    def signature(self):
        return "zh"

    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    @staticmethod
    @lru_cache(maxsize=2**16)
    def _is_chinese_char(uchar):
        """
        :param uchar: input char in unicode
        :return: whether the input char is a Chinese character.
        """
        for start, end in _UCODE_RANGES:
            if start <= uchar <= end:
                return True
        return False

    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        """The tokenization of Chinese text in this script contains two
        steps: separate each Chinese characters (by utf-8 encoding); tokenize
        the non Chinese part (following the `13a` i.e. mteval tokenizer).

        Author: Shujian Huang huangsj@nju.edu.cn

        :param line: input sentence
        :return: tokenized sentence
        """

        line = line.strip()
        line_in_chars = ""

        # TODO: the below code could probably be replaced with the following:
        # @ozan: Gives slightly different scores, need to investigate
        # import regex
        # line = regex.sub(r'(\p{Han})', r' \1 ', line)
        for char in line:
            if self._is_chinese_char(char):
                line_in_chars += " "
                line_in_chars += char
                line_in_chars += " "
            else:
                line_in_chars += char

        return self._post_tokenizer(line_in_chars)


def compute_bleu(
    reference_corpus, translation_corpus, max_order=4, smooth=False
):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for references, translation in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (
                possible_matches_by_order[i] + 1.0
            )
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (
                    float(matches_by_order[i]) / possible_matches_by_order[i]
                )
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1 - 1.0 / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i : i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute(
    predictions,
    references,
    lang="en",
    max_order=4,
    smooth=False,
):
    if lang == "en":
        tokenizer = Tokenizer13a()
    elif lang in ["cn", "zh"]:
        tokenizer = TokenizerZh()
    # if only one reference is provided make sure we still use list of lists
    if isinstance(references[0], str):
        references = [[ref] for ref in references]

    references = [[tokenizer(r) for r in ref] for ref in references]
    predictions = [tokenizer(p) for p in predictions]
    score = compute_bleu(
        reference_corpus=references,
        translation_corpus=predictions,
        max_order=max_order,
        smooth=smooth,
    )
    (bleu, precisions, bp, ratio, translation_length, reference_length) = score
    return {
        "bleu": bleu,
        "precisions": precisions,
        "brevity_penalty": bp,
        "length_ratio": ratio,
        "translation_length": translation_length,
        "reference_length": reference_length,
    }

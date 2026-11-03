"""Matching rules - no OCR needed, detections are built by hand."""
import numpy as np

from ocr_engine import Detection, extract_tokens, find_exact, normalize_query, rank_candidates


def det(text, x0=100, y0=50, w=200, h=30, score=0.95):
    poly = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], dtype=np.float32)
    return Detection(text, score, poly)


def test_query_is_normalised_but_never_converted():
    assert normalize_query(" 1234 567 890 ")[0] == "1234567890"
    q, msg = normalize_query("12345OO890")          # letter O
    assert q is None and "digits only" in msg
    q, msg = normalize_query("123456789")           # 9 digits: allowed but flagged
    assert q == "123456789" and "10 digits" in msg
    assert normalize_query("")[0] is None


def test_exact_match_only():
    toks = extract_tokens([det("1234567890"), det("1234567891", y0=100)])
    m = find_exact(toks, "1234567890")
    assert len(m) == 1 and m[0].kind == "exact"
    assert find_exact(toks, "1234567892") == []     # one digit different -> no match


def test_letters_are_never_silently_turned_into_digits():
    toks = extract_tokens([det("12345O7890")])       # OCR read a letter O
    assert find_exact(toks, "1234507890") == []      # must NOT count as exact
    cands = rank_candidates(toks, "1234507890")      # but may be offered for explicit confirmation
    assert cands and cands[0].text == "1234507890"


def test_glued_neighbours_are_split_by_position():
    toks = extract_tokens([det("12345678902222222222", w=400)])
    m = find_exact(toks, "1234567890")
    assert len(m) == 1 and m[0].kind == "exact_in_merged"
    x = m[0].poly[:, 0]
    assert x.min() >= 100 - 1 and x.max() <= 100 + 400 * 0.55      # left half only
    m2 = find_exact(toks, "2222222222")
    assert m2[0].poly[:, 0].min() >= 100 + 400 * 0.45               # right half only


def test_duplicate_numbers_give_two_matches():
    toks = extract_tokens([det("1234567890", y0=50), det("1234567890", y0=400)])
    assert len(find_exact(toks, "1234567890")) == 2

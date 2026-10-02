from collections import Counter
import heapq
import math


def Shannon_entropy(data, alphabet_size=None):
    """
    H(X) = -sum( p_i * log2(p_i) ),  p_i = f_i / N      (bits per symbol)
    H_max = log2(|Sigma|)                               (bits per symbol)
    Normalized = H(X) / H_max * 100 %

    data          : a string (symbols = characters/bytes) or a list of tokens
    alphabet_size : |Sigma|. For a byte string it defaults to 256 (8 bits).
                    For a token list it MUST be given, because the number of
                    distinct tokens seen is NOT the size of the alphabet.
    Returns (entropy, normalized_percent, h_max)
    """
    if not data:                       # empty input: avoid division by zero
        return 0.0, 0.0, 0.0

    # Step 1: absolute frequency f_i of every distinct symbol/token, and N
    frequency = Counter(data)
    length = len(data)

    # Step 2: entropy  H = -sum(p * log2 p)
    entropy = 0.0
    for freq in frequency.values():
        probability = freq / length
        entropy -= probability * math.log2(probability)

    # Step 3: alphabet size |Sigma|.
    # (A-F), which wrongly normalized a byte string against 4 bits.
    if alphabet_size is None:
        if isinstance(data, str):
            alphabet_size = 256        # one character = one byte = 8 bits
        else:
            raise ValueError("Give alphabet_size explicitly for token lists")

    # Step 4: H_max = log2(|Sigma|)
    h_max = math.log2(alphabet_size) if alphabet_size > 1 else 0.0

    # Step 5: normalized entropy (%)
    normalized_efficiency = (entropy / h_max) * 100 if h_max > 0 else 0.0
    return entropy, normalized_efficiency, h_max


def average_code_length(freqs, lengths):
    """
    L_AVG = sum(f_i * l_i) / sum(f_i)
    freqs   : {symbol: frequency f_i}
    lengths : {symbol: code length l_i in bits}
    """
    return sum(freqs[s] * lengths[s] for s in freqs) / sum(freqs.values())


def huffman_lengths(symbols):
    """
    Small helper: Huffman code length (leaf depth) of each symbol.
    Tie-breaking can change the tree but never the total cost, so L_AVG is
    the same however ties are broken. (Full tree/canonical codes: Huffman_coding.py)
    """
    counts = Counter(symbols)
    if len(counts) == 1:                          # single symbol -> 1 bit
        return {next(iter(counts)): 1}
    # heap items: (weight, unique id, {symbol: depth so far})
    heap = [(f, i, {s: 0}) for i, (s, f) in enumerate(counts.items())]
    heapq.heapify(heap)
    uid = len(heap)
    while len(heap) > 1:
        w1, _, d1 = heapq.heappop(heap)           # two lowest weights
        w2, _, d2 = heapq.heappop(heap)
        merged = {s: d + 1 for s, d in {**d1, **d2}.items()}  # one level deeper
        heapq.heappush(heap, (w1 + w2, uid, merged))
        uid += 1
    return heap[0][2]


# ==============================================================================
# BENCHMARK EXECUTION & METRIC COMPARISON
# ==============================================================================
# Input: ASCII byte string (1 character = 1 byte), |Sigma| = 256
sample_data = "ABCCCDEEFFEEEEAAAAABC"

# LZ77 output as DEFLATE codes (produced and verified by decoding in LZ77.py):
#   A B C C C D E E F F E [D=1,L=3] A [D=1,L=4] B C
# Literals = ASCII values, matches = (length code, distance code) tuples.
deflate_tokens = [65, 66, 67, 67, 67, 68, 69, 69, 70, 70, 69,
                  (257, 0), 65, (258, 0), 66, 67]

# 1. Metrics for the original byte string (alphabet = 256 bytes)
h_raw, eff_raw, hmax_raw = Shannon_entropy(sample_data, 256)

# 2. Metrics for the DEFLATE token stream.
#    Nominal alphabet = 286 literal/length codes (0-285) for the baseline only.
h_tok, eff_tok, hmax_tok = Shannon_entropy(deflate_tokens, 286)

# 3. Average code length:
#    raw data  -> fixed 8-bit code for every byte
#    tokens    -> Huffman code lengths of the token stream
raw_freq = Counter(sample_data)
raw_len = {s: 8 for s in raw_freq}
tok_freq = Counter(deflate_tokens)
tok_len = huffman_lengths(deflate_tokens)

print(f"{'Metric':<27} | {'Raw Data':<15} | {'DEFLATE Tokens':<15}")
print("-" * 62)
print(f"{'Stream Length (N)':<27} | {len(sample_data):<15} | {len(deflate_tokens):<15}")
print(f"{'Entropy H(X) (bits/symbol)':<27} | {h_raw:<15.4f} | {h_tok:<15.4f}")
print(f"{'Max Entropy H_max':<27} | {hmax_raw:<15.4f} | {hmax_tok:<15.4f}")
print(f"{'Normalized Entropy':<27} | {eff_raw:<14.2f}% | {eff_tok:<14.2f}%")
print(f"{'Average Code Length':<27} | {average_code_length(raw_freq, raw_len):<15.4f} | {average_code_length(tok_freq, tok_len):<15.4f}")
# NOTE: per-symbol values of different representations are not directly
# comparable. Compression is shown by encoded bit counts (Huffman_coding.py).
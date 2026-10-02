import heapq
from collections import Counter
import math

END_OF_BLOCK = 256        # DEFLATE end-of-block symbol (part of the literal/length alphabet)


class Node:
    """
    Node in the Huffman tree with a deterministic tie-break rule:
      1) lower frequency first
      2) internal nodes before original leaf symbols
      3) lower symbol value / earlier-created node
    """
    def __init__(self, symbol, freq, order, is_internal=False):
        self.symbol = symbol          # DEFLATE integer code (None for internal nodes)
        self.freq = freq              # frequency (leaf) or sum of children (internal)
        self.order = order            # tie-breaker value
        self.is_internal = is_internal
        self.left = None
        self.right = None

    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq < other.freq
        if self.is_internal != other.is_internal:
            return self.is_internal
        return self.order < other.order


def derive_huffman_bit_lengths(tokens):
    """
    Builds the Huffman tree by repeatedly merging the two lowest-weight nodes,
    then returns {symbol: code length}. Code length = depth of the leaf.
    """
    if not tokens:
        return {}
    counts = Counter(tokens)
    heap = []
    for symbol, freq in counts.items():
        heap.append(Node(symbol=symbol, freq=freq, order=symbol, is_internal=False))
    heapq.heapify(heap)

    # Single-symbol edge case: assign a 1-bit code
    if len(heap) == 1:
        return {heap[0].symbol: 1}

    internal_order = 0
    while len(heap) > 1:
        left = heapq.heappop(heap)        # lowest weight
        right = heapq.heappop(heap)       # second lowest
        merged = Node(symbol=None, freq=left.freq + right.freq,
                      order=internal_order, is_internal=True)
        merged.left = left
        merged.right = right
        internal_order += 1
        heapq.heappush(heap, merged)

    root = heap[0]
    bit_lengths = {}

    def walk_tree(node, depth):           # depth of a leaf = its code length
        if node.symbol is not None:
            bit_lengths[node.symbol] = depth
            return
        walk_tree(node.left, depth + 1)
        walk_tree(node.right, depth + 1)

    walk_tree(root, 0)
    return bit_lengths


def generate_canonical_codes(symbol_lengths):
    """
    RFC 1951 canonical Huffman codes from code lengths only.
    The tree fixes the LENGTHS; canonical assignment re-assigns the bit
    patterns so a decoder can rebuild all codes from the lengths alone.
    """
    if not symbol_lengths:
        return {}
    max_bits = max(symbol_lengths.values())

    # Step 1: bl_count[n] = how many codes have length n
    bl_count = [0] * (max_bits + 1)
    for length in symbol_lengths.values():
        if length > 0:
            bl_count[length] += 1

    # Step 2: first code of each length:
    #         next_code[n] = (next_code[n-1] + bl_count[n-1]) << 1
    next_code = [0] * (max_bits + 1)
    code = 0
    for bits in range(1, max_bits + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code

    # Step 3: shorter codes first, then increasing symbol value
    sorted_symbols = sorted(symbol_lengths, key=lambda s: (symbol_lengths[s], s))
    canonical_codes = {}
    for symbol in sorted_symbols:
        length = symbol_lengths[symbol]
        if length > 0:
            canonical_codes[symbol] = f"{next_code[length]:0{length}b}"
            next_code[length] += 1
    return canonical_codes


def separate_deflate_streams(deflate_output):
    """
    Splits the mixed DEFLATE stream into the two Huffman alphabets:
      1) literal/length stream (literals 0-255, length codes 257-285)
      2) distance stream (distance codes 0-29)
    """
    literal_length_stream = []
    distance_stream = []
    for item in deflate_output:
        if isinstance(item, (list, tuple)):       # match: [length code, distance code]
            literal_length_stream.append(item[0])
            distance_stream.append(item[1])
        else:                                     # literal byte
            literal_length_stream.append(item)
    return literal_length_stream, distance_stream


def length_extra_bits(code):
    """Extra bits of a length code (RFC 1951): 257-264 none, 265-284 (code-261)//4, 285 none."""
    return 0 if code <= 264 or code == 285 else (code - 261) // 4


def distance_extra_bits(code):
    """Extra bits of a distance code: 0-3 none, 4-29 -> code//2 - 1."""
    return 0 if code < 4 else code // 2 - 1


def average_code_length(freqs, lengths):
    """L_AVG = sum(f_i * l_i) / sum(f_i)"""
    return sum(freqs[s] * lengths[s] for s in freqs) / sum(freqs.values())


def entropy(freqs):
    """H = -sum(p log2 p) of a frequency table."""
    n = sum(freqs.values())
    return -sum(f / n * math.log2(f / n) for f in freqs.values())


def run_huffman(label, deflate_stream, original_bytes):
    """Full second-stage pipeline for one DEFLATE stream."""
    # Step 1: separate into the two alphabets
    lit_len_stream, dist_stream = separate_deflate_streams(deflate_stream)
    lit_len_stream.append(END_OF_BLOCK)           # Include end-of-block (256)

    print("=" * 65)
    print(label)
    print("=" * 65)

    # Step 2: tree 1 - literal/length
    lit_len_lengths = derive_huffman_bit_lengths(lit_len_stream)
    lit_len_codes = generate_canonical_codes(lit_len_lengths)
    lit_len_freq = Counter(lit_len_stream)
    print("\n1. LITERAL / LENGTH HUFFMAN TREE")
    print("-" * 45)
    print(f"{'Symbol':<10} | {'Frequency':<10} | {'Bit Length':<10} | {'Canonical Code':<15}")
    print("-" * 45)
    for sym in sorted(lit_len_codes, key=lambda s: (lit_len_lengths[s], s)):
        print(f"{sym:<10} | {lit_len_freq[sym]:<10} | {lit_len_lengths[sym]:<10} | {lit_len_codes[sym]:<15}")

    # Step 3: tree 2 - distance
    dist_lengths = derive_huffman_bit_lengths(dist_stream)
    dist_codes = generate_canonical_codes(dist_lengths)
    dist_freq = Counter(dist_stream)
    print("\n2. DISTANCE HUFFMAN TREE")
    print("-" * 45)
    print(f"{'Symbol':<10} | {'Frequency':<10} | {'Bit Length':<10} | {'Canonical Code':<15}")
    print("-" * 45)
    for sym in sorted(dist_codes, key=lambda s: (dist_lengths[s], s)):
        print(f"{sym:<10} | {dist_freq[sym]:<10} | {dist_lengths[sym]:<10} | {dist_codes[sym]:<15}")

    # Step 4: average code length and the Huffman bound  H <= L_AVG < H + 1
    l_avg = average_code_length(lit_len_freq, lit_len_lengths)
    h = entropy(lit_len_freq)
    print(f"\nL_AVG (literal/length) = {l_avg:.4f}   H = {h:.4f}   "
          f"H <= L_AVG < H+1 : {h <= l_avg < h + 1}")

    # Step 5: actual encoded size = Huffman bits + extra bits
    ll_bits = sum(lit_len_freq[s] * lit_len_lengths[s] for s in lit_len_freq)
    d_bits = sum(dist_freq[s] * dist_lengths[s] for s in dist_freq)
    extra = sum(length_extra_bits(c) for c in lit_len_stream if 257 <= c <= 285)
    extra += sum(distance_extra_bits(c) for c in dist_stream)
    total = ll_bits + d_bits + extra
    original = 8 * original_bytes
    print(f"Encoded bits = {ll_bits} (lit/len) + {d_bits} (distance) + {extra} (extra) = {total}")
    print(f"Original = {original} bits, reduction = {(1 - total / original) * 100:.2f}% "
          f"(before block header / code-length overhead)\n")
    return lit_len_codes, total


# =========================================================
# Execution Driver
# =========================================================
# Self-assessment Q2: stream of "ABCCCDEEFFEEEEAAAAABC" (from LZ77.py)
q2_stream = [65, 66, 67, 67, 67, 68, 69, 69, 70, 70, 69,
             [257, 0], 65, [258, 0], 66, 67]
codes, total = run_huffman("SELF-ASSESSMENT Q2 (N = 21 bytes)", q2_stream, 21)
assert total == 53
assert codes == {67: "00", 65: "010", 66: "011", 69: "100", 70: "101",
                 68: "1100", 256: "1101", 257: "1110", 258: "1111"}

# Main example (Activity 2 and 3): stream of the 30-byte string S
main_stream = [65, 65, 66, 66, 67, [259, 4], 67, [262, 5], [260, 5], 65, [258, 0]]
codes, total = run_huffman("MAIN EXAMPLE (N = 30 bytes)", main_stream, 30)
assert total == 44
assert codes == {65: "00", 67: "01", 66: "100", 262: "101",
                 256: "1100", 258: "1101", 259: "1110", 260: "1111"}
print("All checks passed.")
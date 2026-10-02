def lz77_compress(data: str, window_size: int = 32768, max_match: int = 258, min_match: int = 3):
    """
    Output tokens:
        ("L", char)                 -> literal
        ("M", distance, length)     -> back-reference
    Rule: at each position take the LONGEST match inside the window,
    NEAREST distance on ties, accepted only if length >= 3 (DEFLATE minimum).
    """
    if not data:
        return []
    n = len(data)
    current = 0                                   # position being encoded
    tokens = []

    while current < n:
        max_distance = min(window_size, current)  # how far back we may look
        remaining = n - current
        max_length = min(max_match, remaining)    # DEFLATE caps a match at 258

        best_distance = 0
        best_length = 0

        # Try every start in the window, nearest first (distance 1, 2, 3 ...)
        for distance in range(1, max_distance + 1):
            length = 0
            while length < max_length:
                source_index = current - distance + length
                # Overlapping matches are allowed (distance < length): the
                # source may run into text that is being copied right now.
                if data[source_index] != data[current + length]:
                    break
                length += 1
            # Strict '>' keeps the NEAREST distance when lengths tie
            if length > best_length:
                best_length = length
                best_distance = distance
            if best_length == max_length:         # cannot do better
                break

        if best_length >= min_match:              # long enough -> emit match
            tokens.append(("M", best_distance, best_length))
            current += best_length                # skip the whole matched text
        else:                                     # too short -> emit literal
            tokens.append(("L", data[current]))
            current += 1
    return tokens


def lz77_decompress(tokens):
    """
    NEW: rebuild the original string from tokens. A round-trip check
    (decompress(compress(s)) == s) catches wrong tokens such as the old
    round-11 error, where [D=1, L=4] decoded to FFFF instead of EEEE.
    """
    out = []
    for token in tokens:
        if token[0] == "L":
            out.append(token[1])
        else:
            _, distance, length = token
            for _ in range(length):
                out.append(out[-distance])        # one char at a time: overlap-safe
    return "".join(out)


def get_length_code(length: int):
    """
    Match length (3..258) -> (length code 257..285, extra bits, extra value).
    The code identifies a range; the extra bits select the exact length:
        length = base length + extra value
    """
    if length < 3 or length > 258:
        raise ValueError(f"Match length {length} out of DEFLATE range (3..258)")
    if 3 <= length <= 10:                         # codes 257-264, no extra bits
        return (257 + (length - 3), 0, 0)
    if length == 258:                             # code 285, no extra bits
        return (285, 0, 0)
    # lengths 11..257: (first code, extra bits, base length, range size per code)
    ranges = [
        (265, 1, 11, 2),
        (269, 2, 19, 4),
        (273, 3, 35, 8),
        (277, 4, 67, 16),
        (281, 5, 131, 32),
    ]
    for start_code, extra_bits, base_len, step in ranges:
        if base_len <= length < base_len + (4 * step):
            code_offset = (length - base_len) // step
            extra_val = (length - base_len) % step
            return (start_code + code_offset, extra_bits, extra_val)


def get_distance_code(distance: int):
    """
    Distance (1..32768) -> (distance code 0..29, extra bits, extra value).
        distance = base distance + extra value
    """
    if distance < 1 or distance > 32768:
        raise ValueError(f"Distance {distance} out of DEFLATE range (1..32768)")
    if 1 <= distance <= 4:                        # codes 0-3, no extra bits
        return (distance - 1, 0, 0)
    # distances 5..32768: (first code, extra bits, base distance, range size)
    ranges = [
        (4, 1, 5, 2),
        (6, 2, 9, 4),
        (8, 3, 17, 8),
        (10, 4, 33, 16),
        (12, 5, 65, 32),
        (14, 6, 129, 64),
        (16, 7, 257, 128),
        (18, 8, 513, 256),
        (20, 9, 1025, 512),
        (22, 10, 2049, 1024),
        (24, 11, 4097, 2048),
        (26, 12, 8193, 4096),
        (28, 13, 16385, 8192),
    ]
    for start_code, extra_bits, base_dist, step in ranges:
        if base_dist <= distance < base_dist + (2 * step):
            code_offset = (distance - base_dist) // step
            extra_val = (distance - base_dist) % step
            return (start_code + code_offset, extra_bits, extra_val)


def format_tokens(tokens):
    """Human-readable tokens: literals as characters, matches as [D=.., L=..]."""
    formatted = []
    for token in tokens:
        if token[0] == "L":
            formatted.append(token[1])
        else:
            _, distance, length = token
            formatted.append(f"[D={distance}, L={length}]")
    return formatted


def deflate_code(tokens):
    """
    LZ77 tokens -> DEFLATE-style numbers.
    Literal -> ASCII value (0..255); match -> [length code, distance code].
    (Extra bits are not listed here; get_*_code returns them when needed.)
    """
    deflate_tokens = []
    for token in tokens:
        if token[0] == "L":
            deflate_tokens.append(ord(token[1]))
        elif token[0] == "M":
            _, distance, length = token
            len_code, _, _ = get_length_code(length)
            dist_code, _, _ = get_distance_code(distance)
            deflate_tokens.append([len_code, dist_code])
    return deflate_tokens


# =========================================================
# Execution Example: Activity 2 string and Self-assessment Q1 string
# =========================================================
for sample_data in ["AABBCAABBCCBCAABBCCCAABBCAAAAA", "ABCCCDEEFFEEEEAAAAABC"]:
    tokens = lz77_compress(sample_data)

    # Verification: decoding must rebuild the original string exactly
    assert lz77_decompress(tokens) == sample_data

    deflate_stream = deflate_code(tokens)
    print("Original Data :", sample_data)
    print("LZ77 Tokens   :", " ".join(format_tokens(tokens)))
    print("DEFLATE-style Numeric Stream :", ", ".join(str(i) for i in deflate_stream))
    print()
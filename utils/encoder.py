"""Char pair encoding utilities"""
import unicodedata
from functools import lru_cache


pre_define_token = [
    "<UNK>", "<SEP>", "<BOS>", "<EOS>",
    "<P>", "<S>", "<N>", "<Z>", "<C>"
 ]


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent
    coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1)) \
        + list(range(ord("¡"), ord("¬")+1)) \
        + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_norm_text(text):
    return unicodedata.normalize("NFC", text)


def char_to_id(filepath, pre_define_token, char_number=5000):
    """
    Args:
        filepath: string, char file path, each char split with "\n".
        char_number: int, take first number rows in char file.

    Return single char to id mapping table.
    """
    # Load char
    char_set = set()
    with open(filepath, "r") as f:
        c_list = f.read().split("\n")

    for c in c_list:
        char_set.add(c)
        if len(char_set) >= char_number:
            break

    char_list = sorted(list(char_set))

    tokens = pre_define_token + char_list
    ids = list(range(len(tokens)))

    return dict(zip(tokens, ids))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace', char_use=False):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        if char_use:
            self.pre_define_token = pre_define_token
        else:
            self.byte_encoder = bytes_to_unicode()
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.char_use = char_use

    def bpe(self, tuple_str):
        pairs = get_pairs(tuple_str)

        if not pairs:
            return tuple_str

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_tuple_str = []
            i = 0
            while i < len(tuple_str):
                try:
                    j = tuple_str.index(first, i)
                    new_tuple_str.extend(tuple_str[i:j])
                    i = j
                except ValueError:
                    new_tuple_str.extend(tuple_str[i:])
                    break

                if tuple_str[i] == first and i < len(tuple_str)-1 and tuple_str[i+1] == second:
                    new_tuple_str.append(first+second)
                    i += 2
                else:
                    new_tuple_str.append(tuple_str[i])
                    i += 1
            new_tuple_str = tuple(new_tuple_str)
            tuple_str = new_tuple_str
            if len(tuple_str) == 1:
                break
            else:
                pairs = get_pairs(tuple_str)

        return tuple_str

    def encode(self, text):
        if not text:
            return None
        if self.char_use:
            return self._encode_char(text)
        else:
            return self._encode(text)

    def _encode(self, text):
        text_encode_tuple = tuple(self.byte_encoder[b] for b in text.encode('utf-8'))
        bpe_merged_tuple = self.bpe(text_encode_tuple)
        bpe_token_ids = [self.encoder[bpe_token] for bpe_token in bpe_merged_tuple]

        return bpe_token_ids

    def _encode_char(self, text):
        norm_text = get_norm_text(text)
        char_tuple = tuple(c for c in norm_text)
        bpe_merged_tuple = self.bpe(char_tuple)
        # bpe_token_ids = [self.encoder[bpe_token] for bpe_token in bpe_merged_tuple]
        # Catch unknown char
        bpe_token_ids = []
        for bpe_token in bpe_merged_tuple:
            token_id = self.encoder.get(bpe_token, None)
            if token_id is None:
                cate = unicodedata.category(bpe_token)
                if cate.startswith("P"):
                    token_id = self.encoder["<P>"]
                elif cate.startswith("S"):
                    token_id = self.encoder["<S>"]
                elif cate.startswith("N"):
                    token_id = self.encoder["<N>"]
                elif cate.startswith("Z"):
                    token_id = self.encoder["<Z>"]
                elif cate.startswith("C"):
                    token_id = self.encoder["<C>"]
                else:
                    token_id = self.encoder["<UNK>"]

            bpe_token_ids.append(token_id)

        return bpe_token_ids

    def decode(self, tokens):
        if not tokens:
            return None
        text = ''.join([self.decoder[token] for token in tokens])
        if not self.char_use:
            text = bytearray(
                [self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)

        return text


def get_encoder(bpe_path, merge_num=50000, char_path=None):
    encoder = {}
    if char_path:
        char_encoder = char_to_id(char_path, pre_define_token)
        encoder.update(char_encoder)
        char_use = True
    else:
        byte_encoder = bytes_to_unicode()
        for idx in byte_encoder:
            token = byte_encoder[idx]
            encoder[token] = idx
        char_use = False

    idx = len(encoder)
    bpe_merges = []
    bpe_list = []

    with open(bpe_path, 'r', encoding="utf-8") as f:
        for line in f:
            first, second = line.strip("\n").split(" ")
            bpe_list.append((first, second))

    for i, (first, second) in enumerate(bpe_list):
        if idx >= merge_num:
            break
        if first+second in encoder:
            continue
        encoder[first+second] = idx
        idx += 1
        bpe_merges.append((first, second))

    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
        char_use=char_use
     )

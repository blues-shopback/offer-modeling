import tensorflow as tf


def combine_input(title_enc, desc_enc, inp_len=256, BOS_id=50000, EOS_id=50001,
                  SEP_id=50002, PAD_id=50001, cate_enc=None):
    if cate_enc is not None:
        combined = [BOS_id] + title_enc + [SEP_id] + cate_enc + [SEP_id] + desc_enc + [EOS_id]
        combined = combined[:inp_len]

        cate_pos_token = (
            [0]
            + [0] * len(title_enc) + [0]  # title
            + [1] * len(cate_enc) + [1]  # category
            + [2] * len(desc_enc) + [2]  # description
        )
        cate_pos_token = cate_pos_token[:inp_len]

    else:
        combined = [BOS_id] + title_enc + [SEP_id] + desc_enc + [EOS_id]
        combined = combined[:inp_len]

        cate_pos_token = [0] + [0] * len(title_enc) + [0] + [1] * len(desc_enc) + [1]
        cate_pos_token = cate_pos_token[:inp_len]

    if len(combined) < inp_len:
        pad_len = inp_len - len(combined)
        combined_pad = combined + [PAD_id] * pad_len
        if cate_enc is None:
            cate_pad_id = 2
        else:
            cate_pad_id = 3
        cate_pos_token_pad = cate_pos_token + [cate_pad_id] * pad_len
        attn_mask = [0] * len(combined) + [1] * pad_len
    else:
        combined_pad = combined
        cate_pos_token_pad = cate_pos_token
        attn_mask = [0] * len(combined)

    return combined_pad, cate_pos_token_pad, attn_mask


def encode_and_combine(encoder, title, desc,
                       inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002, PAD_id=50001,
                       cate_list=None):

    title_enc = encoder.encode(title)
    desc_enc = encoder.encode(desc)

    if title_enc is None:
        title_enc = []
    if desc_enc is None:
        desc_enc = []

    if cate_list is not None:
        cate_enc = []
        for cate in cate_list:
            _cate_enc = encoder.encode(cate)
            if _cate_enc:
                cate_enc += _cate_enc
    else:
        cate_enc = None

    combined_pad, cate_pos_token_pad, attn_mask = combine_input(
        title_enc, desc_enc, inp_len=inp_len, BOS_id=BOS_id, EOS_id=EOS_id,
        SEP_id=SEP_id, PAD_id=PAD_id, cate_enc=cate_enc
    )

    return combined_pad, cate_pos_token_pad, attn_mask


def cossim(v1, v2):
    v1_norm = tf.math.l2_normalize(v1, axis=-1)
    v2_norm = tf.math.l2_normalize(v2, axis=-1)

    score = tf.matmul(v1_norm, v2_norm, transpose_b=True)

    return score


def _build_enc(title_desc_list, enc_fn):
    combined_list = []
    cate_pos_list = []
    attn_mask_list = []

    for title, desc in title_desc_list:
        offer_combined, offer_cate_pos, offer_attn_mask = enc_fn(title=title, desc=desc)
        combined_list.append(offer_combined)
        cate_pos_list.append(offer_cate_pos)
        attn_mask_list.append(offer_attn_mask)

    return combined_list, cate_pos_list, attn_mask_list


def _build_enc_with_cate(title_cate_desc_list, enc_fn):
    combined_list = []
    cate_pos_list = []
    attn_mask_list = []

    for title, cate_list, desc in title_cate_desc_list:
        offer_combined, offer_cate_pos, offer_attn_mask = enc_fn(
            title=title, desc=desc, cate_list=cate_list)
        combined_list.append(offer_combined)
        cate_pos_list.append(offer_cate_pos)
        attn_mask_list.append(offer_attn_mask)

    return combined_list, cate_pos_list, attn_mask_list


def eval_pairs(offer_model, enc_fn, pairs):

    title_desc_list = []

    pairs_data = []

    for pair in pairs:
        title_a = pair["title_a"]
        title_b = pair["title_b"]
        desc_a = pair["x_desc1_a"]
        desc_b = pair["x_desc1_b"]
        label = pair["label"]

        pairs_data.append(
            {
                "title_a": title_a,
                "title_b": title_b,
                "desc_a": desc_a,
                "desc_b": desc_b,
                "label": label
            }
        )

        title_desc_list.append((title_a, desc_a))
        title_desc_list.append((title_b, desc_b))

    combined_list, cate_pos_list, attn_mask_list = _build_enc(title_desc_list, enc_fn)
    combined_tensor = tf.constant(combined_list, dtype=tf.int32)
    cate_pos_tensor = tf.constant(cate_pos_list, dtype=tf.int32)
    attn_mask_tensor = tf.constant(attn_mask_list, dtype=tf.float32)
    pooled, output = offer_model(combined_tensor, cate_pos_tensor, attn_mask_tensor)

    pooled_norm = tf.math.l2_normalize(pooled, axis=-1)

    va_list = []
    vb_list = []

    for i in range(0, len(title_desc_list), 2):
        v_a = pooled_norm[i][None]
        v_b = pooled_norm[i+1][None]
        va_list.append(v_a)
        vb_list.append(v_b)

    va = tf.concat(va_list, axis=0)
    vb = tf.concat(vb_list, axis=0)

    scores = tf.einsum("bd,bd->b", va, vb)

    print_str = ""
    for i, pair_info in enumerate(pairs_data):
        score = scores[i]

        s = "=========\n"
        s += "score: {:.4f}".format(score)
        s += "    "
        s += "label: {}\n".format(pair_info["label"])
        s += "title_a: {}\n".format(pair_info["title_a"])
        s += "title_b: {}\n".format(pair_info["title_b"])
        # s += "desc_a: {}\n".format(pair_info["desc_a"])
        # s += "desc_b: {}\n".format(pair_info["desc_b"])
        print_str += s

    return print_str


def eval_query(offer_model, enc_fn, query, offers):
    title_desc_list = []
    title_desc_list.append((query, ""))

    offer_titles = []
    offer_descs = []
    offer_labels = []

    for offer in offers:
        title = offer["title"]
        desc = offer["x_desc1"]
        title_desc_list.append((title, desc))
        offer_titles.append(title)
        offer_descs.append(desc)
        offer_labels.append(offer["label"])

    combined_list, cate_pos_list, attn_mask_list = _build_enc(title_desc_list, enc_fn)
    combined_tensor = tf.constant(combined_list, dtype=tf.int32)
    cate_pos_tensor = tf.constant(cate_pos_list, dtype=tf.int32)
    attn_mask_tensor = tf.constant(attn_mask_list, dtype=tf.float32)

    pooled, output = offer_model(combined_tensor, cate_pos_tensor, attn_mask_tensor)

    pooled_norm = tf.math.l2_normalize(pooled, axis=-1)

    query_norm = pooled_norm[0:1]
    offers_norm = pooled_norm[1:]

    scores = tf.einsum("ij,kj->ik", query_norm, offers_norm)

    # Print result
    print_str = "query: {}\n".format(query)

    for i in range(len(offer_titles)):
        score = scores[0, i]
        title = offer_titles[i]
        desc = offer_descs[i]
        label = offer_labels[i]

        s = "=========\n"
        s += "score: {:.4f}".format(score)
        s += "    "
        s += "label: {}\n".format(label)
        s += "title: {}\n".format(title)
        # s += "desc: {}\n".format(desc)

        print_str += s

    return print_str

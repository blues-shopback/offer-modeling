

def encode_and_combine(encoder, title, desc,
                       inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002, PAD_id=50001,
                       cate=None):

    title_enc = encoder.encode(title)
    desc_enc = encoder.encode(desc)

    if cate is not None:
        cate_enc = encoder.encode(cate)
        combined = [BOS_id] + title_enc + [SEP_id] + cate_enc + [SEP_id] + desc_enc + [EOS_id]
        combined = combined[:inp_len]

        cate_pos_token = [0] + [0] * len(title_enc) + [0] + [1] * len(desc_enc) + [1]
        cate_pos_token = cate_pos_token[:inp_len]

    else:
        combined = [BOS_id] + title_enc + [SEP_id] + desc_enc + [EOS_id]
        combined = combined[:inp_len]

        cate_pos_token = [0] + [0] * len(title_enc) + [0] + [1] * len(desc_enc) + [1]
        cate_pos_token = cate_pos_token[:inp_len]

    if len(combined) < inp_len:
        pad_len = inp_len - len(combined)
        combined += [PAD_id] * pad_len
        cate_pos_token += [2] * pad_len
        attn_mask = [0] * len(combined) + [1] * pad_len

    return combined, cate_pos_token, attn_mask


def encode_and_combine_v2(encoder, title, cate, desc,
                          inp_len=256, BOS_id=50000, EOS_id=50001, SEP_id=50002, PAD_id=50001):

    title_enc = encoder.encode(title)
    cate_enc = encoder.encode(cate)
    desc_enc = encoder.encode(desc)

    combined = [BOS_id] + title_enc + [SEP_id] + cate_enc + [SEP_id] + desc_enc + [EOS_id]
    combined = combined[:inp_len]

    cate_pos_token = [0] + [0] * len(title_enc) + [0] + [1] * len(desc_enc) + [1]
    cate_pos_token = cate_pos_token[:inp_len]

    if len(combined) < inp_len:
        pad_len = inp_len - len(combined)
        combined += [PAD_id] * pad_len
        cate_pos_token += [2] * pad_len
        attn_mask = [0] * len(combined) + [1] * pad_len

    return combined, cate_pos_token, attn_mask

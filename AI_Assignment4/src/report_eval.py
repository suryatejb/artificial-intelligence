import sys
sys.path.append('src')
from ExtractOpinions import ExtractOpinions
from FindSimilarOpinions import FindSimilarOpinions

GT = {
    'service, good': {
        ('service, excellent', 1), ('service, excellent', 2),
        ('service, great', 5), ('service, great', 14),
        ('service, warm', 8), ('service, solid', 13), ('service, good', 20),
        ('waiter, kind', 16), ('waiter, friendly', 17), ('waiter, attentive', 17),
    },
    'service, bad': {
        ('server, rude', 4), ('service, rude', 7), ('service, bad', 9),
        ('service, slow', 19), ('waiter, slow', 15),
    },
    'atmosphere, good': {
        ('atmosphere, nice', 3), ('atmosphere, nice', 4), ('atmosphere, great', 5),
        ('atmosphere, fun', 14), ('feeling, warm', 12), ('ambience, pleasant', 20),
    },
    'food, delicious': {
        ('meal, delicious', 4), ('food, great', 5), ('food, excellent', 6),
        ('food, hearty', 8), ('food, excellent', 13), ('food, fresh', 14),
        ('food, interesting', 14), ('food, satisfying', 16), ('food, fresh', 18),
    },
}


def find_with_scale(finder, query_opinion, scale):
    similar_opinions = {}
    effective_threshold = finder.cosine_sim * scale
    min_val_sim = effective_threshold * 0.30
    min_attr_sim = 0.24

    parts = query_opinion.split(', ', 1)
    if len(parts) != 2:
        return similar_opinions

    query_attr = parts[0].strip().lower()
    query_val = parts[1].strip().lower()
    query_val_polarity = finder._get_polarity(query_val)

    for opinion, review_ids in finder.extracted_opinions.items():
        op_parts = opinion.split(', ', 1)
        if len(op_parts) != 2:
            continue

        op_attr = op_parts[0].strip().lower()
        op_val = op_parts[1].strip().lower()

        norm_query_attr = finder._normalize_attr(query_attr)
        norm_op_attr = finder._normalize_attr(op_attr)

        attr_sim = finder._phrase_sim(norm_query_attr, norm_op_attr)
        val_sim = finder._phrase_sim(query_val, op_val)

        op_val_polarity = finder._get_polarity(op_val)
        if (
            query_val_polarity != 0.0
            and op_val_polarity != 0.0
            and query_val_polarity * op_val_polarity < 0
        ):
            val_sim = 0.0

        if attr_sim < min_attr_sim or val_sim < min_val_sim:
            continue

        overall_sim = (attr_sim + val_sim) / 2.0
        if overall_sim >= effective_threshold:
            similar_opinions[opinion] = review_ids

    pred_set = set()
    for pair, ids in similar_opinions.items():
        for rid in ids:
            pred_set.add((pair, rid))
    return pred_set


if __name__ == '__main__':
    scale = 0.74
    ext = ExtractOpinions()
    with open('data/assign4_reviews.txt', 'r', encoding='utf-8') as f:
        for review_id, line in enumerate(f, start=1):
            ext.extract_pairs(review_id, line)

    finder = FindSimilarOpinions(0.8, ext.extracted_opinions)

    for q, gt_set in GT.items():
        pred_set = find_with_scale(finder, q, scale)
        tp = pred_set & gt_set
        fp = pred_set - gt_set
        fn = gt_set - pred_set
        print('\nQUERY:', q)
        print('TP', len(tp), 'FP', len(fp), 'FN', len(fn))
        if fp:
            print('FP_EXAMPLES:', sorted(fp)[:5])
        if fn:
            print('FN_EXAMPLES:', sorted(fn)[:5])

import sys
sys.path.append('src')
from ExtractOpinions import ExtractOpinions
from FindSimilarOpinions import FindSimilarOpinions

EXPECTED = {
    'service, good': {1, 2, 5, 8, 13, 14, 16, 17, 20},
    'service, bad': {4, 7, 9, 15, 19},
    'atmosphere, good': {3, 4, 5, 12, 14, 20},
    'food, delicious': {4, 5, 6, 8, 13, 14, 16, 18},
}


ext = ExtractOpinions()
with open('data/assign4_reviews.txt', 'r', encoding='utf-8') as f:
    for rid, line in enumerate(f, start=1):
        ext.extract_pairs(rid, line)

finder = FindSimilarOpinions(0.8, ext.extracted_opinions)

for query, expected_ids in EXPECTED.items():
    result = finder.findSimilarOpinions(query)
    got_ids = set()
    for _, ids in result.items():
        got_ids.update(ids)

    tp = len(got_ids & expected_ids)
    fn = len(expected_ids - got_ids)
    fp = len(got_ids - expected_ids)

    print(f'{query}')
    print(f'  expected: {sorted(expected_ids)}')
    print(f'  got     : {sorted(got_ids)}')
    print(f'  TP={tp}, FN={fn}, FP={fp}')

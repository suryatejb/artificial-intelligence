# This is for INFSCI 2440 in Spring 2026
# Please add comments along with your code
# Task 2: Build Bayesian Network probability distribution from data 

def get_p_b_cd():
    # Compute the conditional probability table P(b | c, d) by counting
    # occurrences in the dataset.
    # b: 3 values (1,2,3)  c: 3 values (1,2,3)  d: 2 values (1,2)
    # Array layout: count_bcd[b_idx][c_idx][d_idx]  (0-based indices)

    # Initialise count and probability tables as nested lists
    count_bcd = [[[0] * 2 for _ in range(3)] for _ in range(3)]

    with open(data_add, 'r') as f:
        next(f)  # skip header line
        for line in f:
            parts = line.strip().split('\t')
            b = int(parts[2]) - 1   # convert 1-based to 0-based
            c = int(parts[3]) - 1
            d = int(parts[4]) - 1
            count_bcd[b][c][d] += 1

    # Convert counts to conditional probabilities P(b | c, d)
    p_b_cd = [[[0.0] * 2 for _ in range(3)] for _ in range(3)]
    for c in range(3):
        for d in range(2):
            total = sum(count_bcd[b][c][d] for b in range(3))
            for b in range(3):
                p_b_cd[b][c][d] = count_bcd[b][c][d] / total if total > 0 else 0.0

    return p_b_cd


def get_p_a_be():
    # Compute the conditional probability table P(a | b, e) by counting
    # occurrences in the dataset.
    # a: 2 values (1,2)  b: 3 values (1,2,3)  e: 2 values (1,2)
    # Array layout: count_abe[a_idx][b_idx][e_idx]  (0-based indices)

    count_abe = [[[0] * 2 for _ in range(3)] for _ in range(2)]

    with open(data_add, 'r') as f:
        next(f)  # skip header line
        for line in f:
            parts = line.strip().split('\t')
            a = int(parts[1]) - 1   # convert 1-based to 0-based
            b = int(parts[2]) - 1
            e = int(parts[5]) - 1
            count_abe[a][b][e] += 1

    # Convert counts to conditional probabilities P(a | b, e)
    p_a_be = [[[0.0] * 2 for _ in range(3)] for _ in range(2)]
    for b in range(3):
        for e in range(2):
            total = sum(count_abe[a][b][e] for a in range(2))
            for a in range(2):
                p_a_be[a][b][e] = count_abe[a][b][e] / total if total > 0 else 0.0

    return p_a_be


# following lines are main function:
data_add = "data//assign2_BNdata.txt"

print()
print("=== Task 2: Bayesian Network ===")
print()
# probability distribution of b.
p_b_cd=get_p_b_cd()
for c in range(3):
    for d in range(2):
        for b in range(3):
            print("P(b=" + str(b+1) + "|c=" + str(c+1) + ",d=" + str(d+1) + ")=" + str(p_b_cd[b][c][d]))


# probability distribution of a.
p_a_be=get_p_a_be()
for b in range(3):
    for e in range(2):
        for a in range(2):
            print("P(a=" + str(a+1) + "|b=" + str(b+1) + ",e=" + str(e+1) + ")=" + str(p_a_be[a][b][e]))


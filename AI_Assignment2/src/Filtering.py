# This is for INFSCI 2440 in Spring 2026
# Please add comments along with your code
# Task 1.a: Filtering inference on HMM

def filtering(evidence_data_add, prior, total_day):
    # Forward algorithm for HMM filtering.
    # Computes P(X_t | e_{1:t}) for each day t from 1 to total_day.

    # Transition model T[new][old]: P(X_t | X_{t-1})
    # States: index 0 = rain, index 1 = sunny
    T = [[0.7, 0.3],   # P(rain|rain)=0.7,  P(rain|sunny)=0.3
         [0.3, 0.7]]   # P(sunny|rain)=0.3, P(sunny|sunny)=0.7

    # Sensor model S[state][obs]: P(E_t | X_t)
    # Observations: index 0 = take umbrella, index 1 = no umbrella
    S = [[0.9, 0.1],   # P(umbrella|rain)=0.9,  P(no umbrella|rain)=0.1
         [0.2, 0.8]]   # P(umbrella|sunny)=0.2, P(no umbrella|sunny)=0.8

    # Read evidence from file; encode umbrella=0, no umbrella=1
    evidence = []
    with open(evidence_data_add, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if 'take umbrella' in parts[1]:
                evidence.append(0)
            else:
                evidence.append(1)

    x_prob_rain = []
    # x_prob_sunny[i] = 1 - x_prob_rain[i]

    # Initialise belief with prior P(X_0)
    f = [prior[0], prior[1]]  # f[0]=P(rain), f[1]=P(sunny)

    for t in range(total_day):
        e = evidence[t]

        # Prediction step: P(X_t) = sum_{x_{t-1}} P(X_t|x_{t-1}) * P(x_{t-1})
        pred_rain  = T[0][0] * f[0] + T[0][1] * f[1]
        pred_sunny = T[1][0] * f[0] + T[1][1] * f[1]

        # Update step: weight by sensor model
        updated_rain  = S[0][e] * pred_rain
        updated_sunny = S[1][e] * pred_sunny

        # Normalise so probabilities sum to 1
        total = updated_rain + updated_sunny
        f = [updated_rain / total, updated_sunny / total]

        x_prob_rain.append(f[0])

    return x_prob_rain




# following lines are main function:
evidence_data_add = "data//assign2_umbrella.txt"
total_day = 100
# the prior distribution on the initial state, P(X0). 50% rainy, and 50% sunny on day 0.
prior = [0.5, 0.5]

x_prob_rain=filtering(evidence_data_add, prior, total_day)
for i in range(100):
    print("Day " + str(i+1) + ": rain " + str(x_prob_rain[i]) + ", sunny " + str(1 - x_prob_rain[i]))

print("*"*100)
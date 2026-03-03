# This is for INFSCI 2440 in Spring 2026
# Please add comments along with your code
# Task 1.b: Prediction inference on HMM

def prediction(evidence_data_add, prior, start_day, end_day):
    # Prediction inference on HMM.
    # Uses the filtered distribution at day 100 and then applies the transition
    # model repeatedly (no new evidence), yielding P(X_t | e_{1:100}) for
    # t = start_day ... end_day.

    # Transition model T[new][old]: P(X_t | X_{t-1})
    # States: index 0 = rain, index 1 = sunny
    T = [[0.7, 0.3],   # P(rain|rain)=0.7,  P(rain|sunny)=0.3
         [0.3, 0.7]]   # P(sunny|rain)=0.3, P(sunny|sunny)=0.7

    # Sensor model S[state][obs]: P(E_t | X_t)
    # Observations: index 0 = take umbrella, index 1 = no umbrella
    S = [[0.9, 0.1],
         [0.2, 0.8]]

    # Filter through all 100 observed days
    # Read evidence (days 1-100)
    evidence = []
    with open(evidence_data_add, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if 'take umbrella' in parts[1]:
                evidence.append(0)
            else:
                evidence.append(1)

    # Run forward algorithm to reach f_{1:100}
    f = [prior[0], prior[1]]   # belief at day 0
    for t in range(100):
        e = evidence[t]
        pred_rain  = T[0][0] * f[0] + T[0][1] * f[1]
        pred_sunny = T[1][0] * f[0] + T[1][1] * f[1]
        updated_rain  = S[0][e] * pred_rain
        updated_sunny = S[1][e] * pred_sunny
        total = updated_rain + updated_sunny
        f = [updated_rain / total, updated_sunny / total]
    # f is now P(X_100 | e_{1:100})

    # Predict future days without evidence
    x_prob_rain = []
    # x_prob_sunny[i] = 1 - x_prob_rain[i]

    for t in range(start_day, end_day + 1):
        # Apply transition model only (no observation to condition on)
        pred_rain  = T[0][0] * f[0] + T[0][1] * f[1]
        pred_sunny = T[1][0] * f[0] + T[1][1] * f[1]
        f = [pred_rain, pred_sunny]
        x_prob_rain.append(f[0])

    return x_prob_rain




# following lines are main function:
evidence_data_add = "data//assign2_umbrella.txt"
start_day = 101
end_day = 150
# the prior distribution on the initial state, P(X0). 50% rainy, and 50% sunny on day 0.
prior = [0.5, 0.5]

print()
print("=== Task 1b: Prediction (Days 101-150) ===")
print()
x_prob_rain=prediction(evidence_data_add, prior, start_day, end_day)
for i in range(start_day, end_day+1):
    print("Day " + str(i) + ": rain " + str(x_prob_rain[i-start_day]) + ", sunny " + str(1 - x_prob_rain[i-start_day]))
    # print("Day " + str(i+1) + ": rain " + str(x_prob_rain[i]) + ", sunny " + str(1 - x_prob_rain[i]))

print("*"*100)
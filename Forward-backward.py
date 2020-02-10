import numpy as np

# Page 597 in the book:
# Transition model
T = np.matrix("0.7 0.3;0.3 0.7")

# Sensor model
O = [np.matrix("0.1 0;0 0.8"), np.matrix("0.9 0;0 0.2")]


# Probability for rain day 0.
start = np.matrix("0.5;0.5")

# Days with umbrella
umbrella = [1, 1, 0, 1, 1]

# Finding the probabilities using the forward algorithm.
# The form of the output is:
# Probability day  1 is:  [[0.81818182]] [[0.18181818]].
# [0.81818182] represents P(Rain_day).
# [0.18181818] represents P(Shines_day).


def taskB():

    day = 1
    probability = start
    # Finding the probabilites when using an umbrella or not.
    # Iterates over the days.
    for i in umbrella:
        probability = forward(i, probability)
        print('Probability day ', day,  'is: ',
              probability[0], probability[1], '\n')
        day += 1


def forward(umbrella, input):
    # equation 15.12 in the book
    gg = np.dot(np.dot(O[umbrella], np.transpose(T)), input)
    return normalize(gg)

# Normalizes the matrix


def normalize(matrix):
    return matrix/sum(matrix)


"""
fv[0] ← prior
for i = 1 to t do
fv[i] ← FORWARD(fv[i − 1], ev[i])
for i = t downto 1 do
sv[i] ← NORMALIZE(fv[i] × b)
b← BACKWARD(b, ev[i])
return sv
"""


def forwardandbackward():
    days = len(umbrella)
    probability = [np.matrix('0;0') for i in range(days)]
    probability[0] = start
    for i in range(1, days):
        probability[i] = forward(umbrella[i], probability[i-1])
    print(probability)


forwardandbackward()

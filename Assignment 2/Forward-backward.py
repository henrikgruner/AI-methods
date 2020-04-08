import numpy as np

'''
TaskB() refers to task b, duh
forwardandbackward(..) refers to task C.

Run the code and the wanted output in task B and C will be printed.
'''

# Transition model - dynamic
T = np.matrix("0.7 0.3;0.3 0.7")

# Sensor model - observable
O = [np.matrix("0.1 0;0 0.8"), np.matrix("0.9 0;0 0.2")]


# Probability for rain day 0.
start = np.matrix("0.5;0.5")

# Days with umbrella
umbrella = [1, 1, 0, 1, 1]


def taskB():
    print('\n')
    print('Task B:')

    day = 1
    probability = start
    # Finding the probabilites when using an umbrella or not.
    for i in umbrella:
        probability = forward(i, probability)
        print('Probability day ', day,  'is: ',
              probability[0], probability[1], '\n')
        day += 1

    print('\n')
    print('\n')


def forward(umbrella, input):
    # equation 15.12 in the book
    gg = np.dot(np.dot(O[umbrella], np.transpose(T)), input)
    return normalize(gg)

# Normalizes the matrix


def normalize(matrix):
    return matrix/sum(matrix)

# Just for printing


def printResults(b, smooth):

    for i in range(len(b)-1):
        print('Backward probability on day:', len(b)-1-i-1,  'is: ',
              b[i][0], b[i][0], '\n')

    for j in range(len(smooth)):
        print('Probability on day :', len(smooth)-j-1,  ' is: ',
              smooth[j][0][0], smooth[j][1], '\n')


def printF(probability):
    for i in range(0, len(probability)):
        print('Forward Probability on day ', i+1-1,  'is: ',
              probability[i][0], probability[i][1], '\n')


def forwardandbackward():
    print('Task C')

    days = len(umbrella)
    probability = [np.matrix('0;0') for i in range(days+1)]
    probability[0] = start  # Start case
    # Backward and smoothed values
    b = np.matrix('1.0; 1.0')
    sv = np.matrix('0; 0')

    # Just for printing
    bvalues = []
    bvalues.append(b)
    smooth = []

    # Forward calculating
    for i in range(1, days+1):
        probability[i] = forward(umbrella[i-1], probability[i-1])

    # Just for printing
    printF(probability)

    # Calculating backwards probabilities and smoothed probabilites.
    for j in range(days, -1, -1):

        sv = normalize(np.multiply(probability[j], b))
        b = backward(umbrella[j-1], b)

        # Just for printing
        smooth.append(sv)
        bvalues.append(b)
    # More printing
    printResults(bvalues, smooth)

# Calculating the backward probabilitites (normalized)


def backward(index, b):
    return normalize(np.dot(np.dot(T, O[index]), b))

    # bk+1:t = TOk+1bk+2:t .


# Task B
taskB()
# Task C
forwardandbackward()

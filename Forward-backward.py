
def forward():
    return None


def backward():
    return None


def normalize(matrix):
    matrixsum = sum(matrix)

    for i in range(matrix):
        matrix[i] = matrix[i]/matrixsum
    return matrix

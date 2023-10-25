import numpy as np
import argparse
from numpy.linalg import norm

global LOG
LOG = False

def log(values, LOG):
    if LOG:
        print(values)


def ones(arr: np.ndarray):
    # Find the indices where value is 1
    y_indices, x_indices = np.where(arr[0:, :abs(arr.shape[1] - arr.shape[0])] == 1)

    # Filter the indices
    valid_indices = [(x, y) for x, y in zip(x_indices, y_indices) if arr[0][x] == 0]
    return valid_indices


def output(tableau: np.ndarray):
    valid_indices = ones(tableau)
    n, m = tableau.shape
    x = np.zeros(abs(m - n))
    for item in valid_indices:
        x[item[0]] = tableau[item[1]][m - 1]
    print("A vector of decision variables:", x)
    print("Optimal value:", np.array([tableau[0][tableau.shape[1] - 1]]))

def initial_X(A: np.ndarray, b: np.ndarray):
    #TODO: write initialization of X
    ...

# Implementation of interior point method
def maximize(A: np.ndarray, b: np.ndarray, c: np.ndarray, accuracy):
    # TODO: x = initial_X(A, b)
    x = np.array([1 / 2, 7 / 2, 1, 2], float)
    n = np.size(A, 1)
    alpha = 0.5

    i = 1
    while True:
        v = x
        D = np.diag(x)
        AA = np.dot(A, D)
        cc = np.dot(D, c)
        I = np.eye(n)
        F = np.dot(AA, np.transpose(AA))
        FI = np.linalg.inv(F)
        H = np.dot(np.transpose(AA), FI)
        P = np.subtract(I, np.dot(H, AA))
        cp = np.dot(P, cc)
        nu = np.absolute(np.min(cp))
        y = np.add(np.ones(n, float), (alpha / nu) * cp)
        x = np.dot(D, y)

        if LOG == True:
            print("In iteration  ", i, " we have x = ", x, "\n")
            i = i + 1

        if norm(np.subtract(x, v), ord=2) < 0.01:
            break

    print("Final X is: ", x)
    print("Final value is: ", np.dot(c, x))


def main():
    global LOG
    input_file = "input.txt"

    task_type = "Maximize"
    ACCURACY = 0.001

    # Create a parser
    parser = argparse.ArgumentParser(description='Simplex method implementation to solve linear programming problems')
    # Add arguments
    parser.add_argument('-i', '--input', type=str, help='Set input file path. Default: input.txt')

    parser.add_argument('-l', '--log', action='store_true', help='Set logging. Default: False')

    parser.add_argument('-p', '--problem', type=str, help='Set problem type. Possible values: min, max. Default: max')

    parser.add_argument('-a', '--accuracy', type=str, help='Set approximation accuracy. Default: 0.001')
    # Parse the arguments
    args = parser.parse_args()

    # Check if input file is provided
    if args.input:
        input_file = args.input
    if args.log:
        LOG = True
    if args.problem == "min":
        task_type = "Minimize"
    elif args.problem == "max":
        ...
    elif args.problem is None:
        task_type = "Maximize"
    else:
        raise ValueError("Problem type can be only min or max")

    if args.accuracy:
        ACCURACY = args.accuracy

    # Read input file
    with open(input_file) as file:
        elements = file.read().replace("\n", ' ').split("#")[1:]
        c, a, b = [np.asarray(i.split(":")[1].strip().split(" ")) for i in elements]
        c = np.array(c, dtype=float)
        b = np.array(b, dtype=float)
        a = np.reshape(np.array(a, dtype=float), (-1, c.size))

        # Change minimize problem to maximize
        # TODO: fix the problem with minimize
        c = -c if task_type == "Minimize" else c

        #np.set_printoptions(precision=decimals, suppress=True)

        if np.any(b < 0):
            log('The method is not applicable!', LOG)
        else:
            maximize(a, b, c, ACCURACY)


if __name__ == '__main__':
    main()

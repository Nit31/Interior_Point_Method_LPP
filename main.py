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

# Implementation of interior point method
def maximize(A: np.ndarray, b: np.ndarray, c: np.ndarray, x: np.ndarray, accuracy):
    # TODO: x = initial_X(A, b)
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

        if LOG:
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
        c, a, b, init_x = [np.asarray(i.split(":")[1].strip().split(" ")) for i in elements]

        # Save signs of inequalities in the list
        signs = []
        i = 0
        while i < len(a):
            if a[i] == "<" or a[i] == ">":
                signs.append(a[i])
                # remove element from a
                a = np.delete(a, i)
                i-=1
            i+=1

        # Convert to normal representation
        c = np.array(c, dtype=float)
        b = np.array(b, dtype=float)
        a = np.reshape(np.array(a, dtype=float), (-1, c.size))
        init_x = np.array(init_x, dtype=float)

        # Change minimize problem to maximize
        # TODO: fix the problem with minimize
        c = -c if task_type == "Minimize" else c

        #np.set_printoptions(precision=decimals, suppress=True)

        # Add slacks to the matrix
        a_shape_slacks = (a.shape[0], a.shape[1] + a.shape[0])
        zeros = np.zeros(a_shape_slacks)
        zeros[:a.shape[0], :a.shape[1]] = a
        a = zeros
        for i in range(a.shape[0]):
            if signs[i] == "<":
                a[i][a.shape[1] - a.shape[0] + i] = 1
            elif signs[i] == ">":
                a[i][a.shape[1] - a.shape[0] + i] = -1

        # Add slacks to the objective function
        zeros = np.zeros(c.shape[0] + a.shape[0])
        zeros[:c.shape[0]] = c
        c = zeros
        if np.any(b < 0):
            raise ValueError("Right-hand side values of the inequality constraints must be non-negative")
        else:
            maximize(a, b, c, init_x, ACCURACY)


if __name__ == '__main__':
    main()

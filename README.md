# Linear program solver

<!-- ABOUT THE PROJECT -->
## About The Project
Interior point method for solving LPP.

### Built With

* [Python](https://www.python.org/)

<!-- GETTING STARTED -->
## Installation
In terminal run the following:

    git clone git@github.com:Nit31/Interior_Point_Method_LPP.git
    cd Interior_Point_Method_LPP
    pip install -r requirements.txt


<!-- USAGE EXAMPLES -->
## Usage
Input format:
The input contains:
* A vector of coefficients of objective function - C.
* A matrix of coefficients of constraint function with the signs of equations - A.
* A vector of right-hand side numbers - b.
* A vector of the initial point lay in the feasible region - x0.

Input Example:
For the problem\
$z = x_1 + x_2$\
$2x_1 + 4x_2 < 16$\
$x_1 + 3x_2 > 9$\
the input.txt file should look like this:

    #Objective function:
    1 1
    #Matrix:
    2 4 <
    1 3 >
    #Vector b:
    16 9
    #Initial x:
    0.5 3.5 1 2

Running the code:

    python main.py -p max -i input.txt

Output format:
* A vector of decision variables - x
* Maximum (minimum) value of the objective function.

Example output:
    
    Final X is:  [5.99360088e+00 1.00274248e+00 1.82831024e-03 1.82832957e-03]
    Final value is:  6.996343360206015

## Options

-h, --help            show this help message and exit\
-i INPUT, --input INPUT\
                    Set input file path. Default: input.txt\
-l, --log             Set logging. Default: False\
-p PROBLEM, --problem PROBLEM\
                    Set problem type. Possible values: min, max. Default: max\
-a ACCURACY, --accuracy ACCURACY\
                    Set approximation accuracy. Default: 0.001

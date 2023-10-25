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
* A vector of coefficients of objective function (including slacks and surpluses) - C.
* A matrix of coefficients of constraint function after adding slack and surplus variables - A.
* A vector of right-hand side numbers - b.

Input Example:
For the problem\
z = x1 + x2\
2x1 + 4x2 < 16\
x1 + 3x2 > 9\
the input.txt file should look like this:

    #Objective function:
    1 1 0 0
    #Matrix:
    2 4 1 0
    1 3 0 -1
    #Vector b:
    16 9

Output format:
* A vector of decision variables - x
* Maximum (minimum) value of the objective function.

Example output:
    
    Final X is:  [5.99360088e+00 1.00274248e+00 1.82831024e-03 1.82832957e-03]
    Final value is:  6.996343360206015


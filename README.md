# Linear program solver

<!-- ABOUT THE PROJECT -->
## About The Project
Interior point method for solving LPP.

### Built With

* [Python](https://www.python.org/)

<!-- GETTING STARTED -->
## Installation
In the terminal run the following:

    git clone 
    pip install -r requirements.txt
    python main.py

### Further reading
* [Simplex method](https://en.wikipedia.org/wiki/Simplex_algorithm)


<!-- USAGE EXAMPLES -->
## Usage
Input format:
The input contains:
* A vector of coefficients of objective function (including slacks and surpluses) - C.
* A matrix of coefficients of constraint function after adding slack and surplus variables - A.
* A vector of right-hand side numbers - b.

Input Example:

    #Objective function:
    1 1 0 0
    #Matrix:
    2 4 1 0
    1 3 0 -1
    #Vector b:
    16 9

Output format:
* The string ”The method is not applicable!”

or
* A vector of decision variables - x
* Maximum (minimum) value of the objective function.

Output of the Example:
    
    Final X is:  [5.99360088e+00 1.00274248e+00 1.82831024e-03 1.82832957e-03]
    Final value is:  6.996343360206015


from colors import bcolors
from matrix_utility import row_addition_elementary_matrix, scalar_multiplication_elementary_matrix
from matrix_utility import partial_pivoting
import numpy as np
"""
date:19/2/24
group members: (1) name: Shulamit-mor-yossef. id: 206576977. (2) name: Zohar-monsonego. id: 214067662. (3) name: hodaya-shirazie. id: 213907785.
submitted by: name: Hodaya-shirazie. id: 213907785.
input: 
output:
"""

"""
Function that find the inverse of non-singular matrix
The function performs elementary row operations to transform it into the identity matrix. 
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
 If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""


def inverse(matrix):
    print(bcolors.OKBLUE, f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n", bcolors.ENDC)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    for i in range(n):

        if matrix[i, i] == 0:
            print(bcolors.YELLOW, "do pivoting------------------------------------------------------------------------------------------------------------", bcolors.ENDC)
            matrix, identity = partial_pivoting(matrix, i, n, identity)
            print(bcolors.OKGREEN,
                  "------------------------------------------------------------------------------------------------------------------",
                  bcolors.ENDC)
            print("identity matrix after pivoting: \n" + str(identity))
            print(bcolors.OKGREEN,
                  "------------------------------------------------------------------------------------------------------------------",
                  bcolors.ENDC)



        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")
            matrix = np.dot(elementary_matrix, matrix)
            print(f"The matrix after elementary operation :\n {matrix}")
            print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",  bcolors.ENDC)
            identity = np.dot(elementary_matrix, identity)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",
                      bcolors.ENDC)
                identity = np.dot(elementary_matrix, identity)
        print("identity matrix: \n" + str(identity))
        print(bcolors.OKGREEN,
              "------------------------------------------------------------------------------------------------------------------",
              bcolors.ENDC)

    return identity


if __name__ == '__main__':

    A_b = np.array([[8, 1, 9],
                 [0, 0, 3],
                    [1, 0, 0]])

    try:
        A_inverse = inverse(A_b)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print("=====================================================================================================================", bcolors.ENDC)

    except ValueError as e:
        print(str(e))



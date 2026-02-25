#!/usr/bin/env python3
"""
Script: thomas_solve_tridiagonal_matrix.py
Objective: Solve tridiagonal linear systems using the Thomas algorithm for SWEB numerical routines.
Author: Yi Yu
Created: 2026-02-17
Last updated: 2026-02-25
Inputs: Lower/main/upper diagonal arrays and right-hand-side vector for a tridiagonal system.
Outputs: Solution vector for the supplied tridiagonal linear system.
Usage: Imported by hydraulic solver modules; not intended as a standalone CLI script.
Dependencies: numpy
"""
# Tridiagonal matrix solver (Thomas algorithm)
def thomas_solve_tridiagonal_matrix(a, b, c, d):
    """
    Solve a tridiagonal matrix system Ax = d
    
    Similar to MatrixSolverTriDiagonal used in the Fortran code

    INVERT (SOLVE) THE TRI-DIAGONAL MATRIX PROBLEM SHOWN BELOW:
    ----------------------------------------------------------------------
    ###                                            ### ###  ###   ###  ###
    #B(1), C(1),  0  ,  0  ,  0  ,   . . .  ,    0   # #      #   #      #
    #A(2), B(2), C(2),  0  ,  0  ,   . . .  ,    0   # #      #   #      #
    # 0  , A(3), B(3), C(3),  0  ,   . . .  ,    0   # #      #   # D(3) #
    # 0  ,  0  , A(4), B(4), C(4),   . . .  ,    0   # # P(4) #   # D(4) #
    # 0  ,  0  ,  0  , A(5), B(5),   . . .  ,    0   # # P(5) #   # D(5) #
    # .                                          .   # #  .   # = #   .  #
    # .                                          .   # #  .   #   #   .  #
    # .                                          .   # #  .   #   #   .  #
    # 0  , . . . , 0 , A(M-2), B(M-2), C(M-2),   0   # #P(M-2)#   #D(M-2)#
    # 0  , . . . , 0 ,   0   , A(M-1), B(M-1), C(M-1)# #P(M-1)#   #D(M-1)#
    # 0  , . . . , 0 ,   0   ,   0   ,  A(M) ,  B(M) # # P(M) #   # D(M) #
    ###                                            ### ###  ###   ###  ###
    ----------------------------------------------------------------------

    Parameters:
    -----------
    a : numpy.ndarray
        Lower diagonal elements
    b : numpy.ndarray
        Main diagonal elements
    c : numpy.ndarray
        Upper diagonal elements
    d : numpy.ndarray
        Right hand side vector
    
    Returns:
    --------
    numpy.ndarray
        Solution vector x
    """
    n = len(d)
    # Make copies to avoid modifying input arrays
    a_c = a.copy()
    b_c = b.copy()
    c_c = c.copy()
    d_c = d.copy()
    x = np.zeros(n)
    
    # Forward elimination
    for i in range(1, n):
        factor = a_c[i] / b_c[i-1]
        b_c[i] = b_c[i] - factor * c_c[i-1]
        d_c[i] = d_c[i] - factor * d_c[i-1]
    
    # Back substitution
    x[n-1] = d_c[n-1] / b_c[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d_c[i] - c_c[i] * x[i+1]) / b_c[i]
    
    return x

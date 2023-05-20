from tkinter import *

import random
import numpy as np
import scipy
import os
def producto_matrices(a, b):
    filas_a = len(a)
    filas_b = len(b)
    columnas_a = len(a[0])
    columnas_b = len(b[0])
    if columnas_a != filas_b:
        return None
    producto = []
    for i in range(filas_b):
        producto.append([])
        for j in range(columnas_b):
            producto[i].append(None)
    for c in range(columnas_b):
        for i in range(filas_a):
            suma = 0
            for j in range(columnas_a):
                suma += a[i][j]*b[j][c]
            producto[i][c] = suma
    return producto

def lu_decomposition(A):
    n = len(A)
    matriz = np.array(A, dtype=float)
    for i in range(n):
        for j in range(n):
            matriz[i][j] = A[i][j]
    n = matriz.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for j in range(n):
        L[j, j] = 1.0
        for i in range(j + 1):
            U[i, j] = matriz[i, j] - np.dot(L[i, :j], U[:j, j])
        for i in range(j + 1, n):
            L[i, j] = (matriz[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]
    return L, U

def escalonar_matriz(A):
    n = len(A)
    matrix = np.array([[0] * n for _ in range(n)])
    for i in range(n):
        for j in range(n):
            matrix[i][j] = A[i][j]
    for i in range(n):
        if matrix[i][i] == 0:
            j = i + 1
            while j < n and matrix[j][i] == 0:
                j += 1
            if j == n:
                return False
            matrix[i], matrix[j] = matrix[j], matrix[i]
        for j in range(i+1, n):
            factor = matrix[j][i] / matrix[i][i]
            for k in range(i, n):
                matrix[j][k] -= factor * matrix[i][k]
    return True

def is_lu_factorizable(A):
    n = len(A)
    matrix = np.array([[0] * n for _ in range(n)])
    if not escalonar_matriz(A):
        return False
    if matrix.shape[0] != matrix.shape[1]:
        return False
    diag = np.abs(np.diag(matrix))
    off_diag = np.sum(np.abs(matrix), axis=1) - diag
    if np.all(diag > off_diag):
        return True
    if np.all(diag >= off_diag):
        return True
    return False

def plu_factorization(A):
    n = len(A)
    matriz = np.array(A, dtype=float)
    for i in range(n):
        for j in range(n):
            matriz[i][j] = A[i][j]
    L = np.eye(n)
    U = matriz.copy()
    P = np.eye(n)
    for j in range(n):
        row = np.argmax(np.abs(U[j:, j])) + j
        if row != j:
            U[[j, row], :] = U[[row, j], :]
            P[[j, row], :] = P[[row, j], :]
            if j > 0:
                L[[j, row], :j] = L[[row, j], :j]
        for i in range(j + 1, n):
            factor = U[i, j] / U[j, j]
            U[i, j:] -= factor * U[j, j:]
            L[i, j] = factor
    return P, L, U


n = int(input("Ingrese un número entre 4 y 10 para el tamaño de la matriz: "))
if n < 4 or n > 10:
    print("El número ingresado no está dentro del rango permitido.")
else:
    A = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            A[i][j] = random.randint(-8, 8)
            A[j][i] = A[i][j]


print("Matriz A:")
for i in range(n):
    for j in range(n):
        print(A[i][j], end=" ")
    print()
if is_lu_factorizable(A):
    print("La matriz es factorizable por LU")
    LU = lu_decomposition(A)
    if LU is not None:
        L, U = LU
        print("Matriz L:")
        for row in L:
            print(row)
        print("Matriz U:")
        for row in U:
            print(row)
        print("L*U:")
        prod = producto_matrices(L,U)
        for row in prod:
            print(row)
else:
    print("La matriz no es factorizable por LU")
    Pt,L,U = plu_factorization(A)
    print("Matriz P^t")
    for row in Pt:
        print(row)
    print("Matriz L:")
    for row in L:
        print(row)
    print("Matriz U:")
    for row in U:
        print(row)
    prodLU = producto_matrices(L, U)
    print("Matriz L*U:")
    for row in prodLU:
        print(row)
    prodAPt = producto_matrices(A, Pt)
    print("Matriz A*Pt:")
    for row in prodLU:
        print(row)

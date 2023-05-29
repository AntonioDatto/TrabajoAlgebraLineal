from urllib.request import urlopen
from tkinter import *
from tkinter import filedialog
import random
import numpy as np
from PIL import Image, ImageTk
import io

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


def factorizeMatrix(event=None):
    n = inputEntry.get()

    if not n:
        resultText.config(state="normal")
        resultText.delete("1.0", END)
        resultText.insert("1.0", "No se ha ingresado ningún valor.")
        resultText.config(state="disabled")
        return

    n = int(n)
    if n < 4 or n > 10:
        inputEntry.delete(0, END)
        resultText.config(state="normal")
        resultText.delete("1.0", END)
        resultText.insert("1.0", "El número ingresado no está dentro del rango permitido.")
        resultText.config(state="disabled")
        return
    else:
        inputEntry.delete(0, END)

        A = [[0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i, n):
                A[i][j] = random.randint(-8, 8)
                A[j][i] = A[i][j]

    resultText.config(state="normal")
    resultText.delete("1.0", END)
    resultText.insert("1.0", "Matriz A:\n")
    for i in range(n):
        for j in range(n):
            resultText.insert(END, str(A[i][j]) + " ")
        resultText.insert(END, "\n")

    if is_lu_factorizable(A):
        resultText.insert(END, "\nLa matriz es factorizable por LU\n\n")

        LU = lu_decomposition(A)
        if LU is not None:
            L, U = LU
            resultText.insert(END, "Matriz L:\n")
            for row in L:
                resultText.insert(END, " ".join([str(val) for val in row]) + "\n")

            resultText.insert(END, "\nMatriz U:\n")
            for row in U:
                resultText.insert(END, " ".join([str(val) for val in row]) + "\n")
                
            resultText.insert(END, "\nMatriz L*U:\n")
            prod = producto_matrices(L, U)
            for row in prod:
                resultText.insert(END, " ".join([str(val) for val in row]) + "\n")
    else:
        resultText.insert(END, "\nLa matriz no es factorizable por LU\n\n")
        Pt, L, U = plu_factorization(A) 
        resultText.insert(END, "Matriz P^t:\n")
        for row in Pt:
            resultText.insert(END, " ".join([str(val) for val in row]) + "\n")
        resultText.insert(END, "\nMatriz L:\n")
        for row in L:
            resultText.insert(END, " ".join([str(val) for val in row]) + "\n")
        resultText.insert(END, "\nMatriz U:\n")
        for row in U:
            resultText.insert(END, " ".join([str(val) for val in row]) + "\n")
        prodLU = producto_matrices(L, U)
        resultText.insert(END, "\nMatriz L*U:\n")
        for row in prodLU:
            resultText.insert(END, " ".join([str(val) for val in row]) + "\n")
        prodPtLU = producto_matrices(Pt, A)
        resultText.insert(END, "\nMatriz P^t*A:\n")
        for row in prodPtLU:
            resultText.insert(END, " ".join([str(val) for val in row]) + "\n")

    resultText.config(state="disabled")

def inicioScreen():
    label_Tema.place(x = 210, y = 120)
    label_Grupo.place(x = 337, y = 184)
    label_Curso.place(x = 280, y = 239)
    Factorizar_Screen.place(x = 350, y = 343)
    Integrantes_Screen.place(x = 350, y = 422)
    salirButton.place(x = 350, y = 502)
    label_datos.place(x = 7, y = 610)

def integranteScreen():

    label_Tema.place_forget()
    label_Grupo.place_forget()
    label_Curso.place_forget()
    Factorizar_Screen.place_forget()
    Integrantes_Screen.place_forget()
    salirButton.place_forget()
    label_datos.place_forget()

    regresarButton.place(x = 10, y = 10)

    label_integrantes.place(x = 120, y = 60)

    label_fotoIam.place(x = 130, y = 190)
    label_Iam.place(x = 110, y = 350)

    label_fotoAntonio.place(x = 640, y = 190)
    label_Antonio.place(x = 610, y = 350)

    label_fotoJhoan.place(x = 210, y = 450)
    label_Jhoan.place(x = 170, y = 600)

    label_fotoLeonardo.place(x = 530, y = 450)
    label_Leonardo.place(x = 490, y = 600)

def factorizacionScreen():
    label_Tema.place_forget()
    label_Grupo.place_forget()
    label_Curso.place_forget()
    Factorizar_Screen.place_forget()
    Integrantes_Screen.place_forget()
    salirButton.place_forget()
    label_datos.place_forget()

    regresarButton.place(x = 10, y = 10)
    inputLabel.pack(pady=15)
    inputEntry.pack()
    inputEntry.focus_set()
    factorizeButton.pack(pady=15)
    resultText.pack()
    guardarButton.place(x = 400, y = 595)

def retornar_Inicio():
    regresarButton.place_forget()
    inputLabel.pack_forget()
    inputEntry.pack_forget()
    factorizeButton.pack_forget()
    resultText.pack_forget()
    guardarButton.place_forget()

    label_integrantes.place_forget()

    label_fotoIam.place_forget()
    label_Iam.place_forget()

    label_fotoAntonio.place_forget()
    label_Antonio.place_forget()

    label_fotoJhoan.place_forget()
    label_Jhoan.place_forget()

    label_fotoLeonardo.place_forget()
    label_Leonardo.place_forget()

    inicioScreen()

def closeWindow():
    window.destroy()

def guardarArchivo():
    archivo = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Archivo de texto", "*.txt")])
    if archivo:
        contenido = resultText.get("1.0", "end-1c")
        with open(archivo, "w") as file:
            file.write(contenido)

def getURL(url):
    with urlopen(url) as response:
        data = response.read()
    image_bytes = io.BytesIO(data)
    return image_bytes

window = Tk()
window.title("Factorización LU")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

x = (screen_width - 900) // 2
y = (screen_height - 650) // 2

window.geometry(f"900x650+{x}+{y}")

window.configure(bg="#1B9C85")

url_logo = "https://i.ibb.co/Z23rYzM/UPC.png"
link_logo = getURL(url_logo)

logoUPC = Image.open(link_logo)
logoUPC_resize = logoUPC.resize((80, 80))

imagenLogoUPC = ImageTk.PhotoImage(logoUPC_resize)

label_logo = Label(window, image=imagenLogoUPC, bg="#1B9C85")
label_logo.place(x = 804, y = 10)

label_Tema = Label(window, text="Factorización LU", fg= "#E8F6EF", bg = "#1B9C85", font=("Lato", 48))

label_Grupo = Label(window, text="Grupo 1", fg= "#E8F6EF", bg = "#1B9C85", font=("Lato", 40))

label_Curso = Label(window, text="Álgebra Lineal", fg= "#E8F6EF", bg = "#1B9C85", font=("Lato", 36))

Factorizar_Screen = Button(window, width=10, command=factorizacionScreen, text="calcular", fg="#000000", bg="#FFE194", font=("Lato", 25))

Integrantes_Screen = Button(window, width=10, command=integranteScreen, text="integrantes", fg="#000000", bg="#E8F6EF", font=("Lato", 25))

salirButton = Button(window, width=10, command = closeWindow, text="salir", fg="#E8F6EF", bg="#4C4C6D", font=("Lato", 25))

label_datos = Label(window, text="CC52 | Nedin Esteban Fernandez Quispe", fg= "#E8F6EF", bg = "#1B9C85", font=("Lato", 20))

label_integrantes = Label(window, text="Integrantes del Grupo", fg= "#E8F6EF", bg = "#1B9C85", font=("Lato", 48))

url_Iam = "https://i.ibb.co/mJSWCLL/Iam.png"
link_Iam = getURL(url_Iam)

fotoIam = Image.open(link_Iam)
fotoIam_resize = fotoIam.resize((130, 140))
imagenIam = ImageTk.PhotoImage(fotoIam_resize)
label_fotoIam = Label(window, image=imagenIam)

label_Iam = Label(window, text="Iam Alvarez", fg= "#E8F6EF", bg = "#1B9C85", font=("Lato", 24))

url_Antonio = "https://i.ibb.co/m9dLB2Z/Antonio.png"
link_Antonio = getURL(url_Antonio)

fotoAntonio = Image.open(link_Antonio)
fotoAntonio_resize = fotoAntonio.resize((130, 140))
imagenAntonio = ImageTk.PhotoImage(fotoAntonio_resize)
label_fotoAntonio = Label(window, image=imagenAntonio)

label_Antonio = Label(window, text="Antonio Datto", fg= "#E8F6EF", bg = "#1B9C85", font=("Lato", 24))

url_Jhoan = "https://i.ibb.co/84ND4hx/Jhoan.png"
link_Jhoan = getURL(url_Jhoan)

fotoJhoan = Image.open(link_Jhoan)
fotoJhoan_resize = fotoJhoan.resize((130, 140))
imagenJhoan = ImageTk.PhotoImage(fotoJhoan_resize)
label_fotoJhoan = Label(window, image=imagenJhoan)

label_Jhoan = Label(window, text="Jhoan Medalla", fg= "#E8F6EF", bg = "#1B9C85", font=("Lato", 24))

url_Leonardo = "https://i.ibb.co/b5R0YfR/Leonardo.png"
link_Leonar = getURL(url_Leonardo)

fotoLeonardo = Image.open(link_Leonar)
fotoLeonardo_resize = fotoLeonardo.resize((130, 140))
imagenLeonardo = ImageTk.PhotoImage(fotoLeonardo_resize)
label_fotoLeonardo = Label(window, image=imagenLeonardo)

label_Leonardo = Label(window, text="Leonardo Bravo", fg= "#E8F6EF", bg = "#1B9C85", font=("Lato", 24))

regresarButton = Button(window, text="🡸", command=retornar_Inicio, fg="#E8F6EF", bg="#1B9C85", font=("Lato", 15))

inputLabel = Label(window, text="Ingrese un número entre 4 y 10 para el tamaño de la matriz:", fg="#E8F6EF", bg="#1B9C85", font=("Lato", 12))

inputEntry = Entry(window)

factorizeButton = Button(window, text="Factorizar", command=factorizeMatrix, fg="#E8F6EF", bg="#4C4C6D", font=("Lato", 12))

window.bind("<Return>", factorizeMatrix)

resultText = Text(window, state="disabled", fg="#000000", bg="#FFE194", font=("Courier New", 12))

guardarButton = Button(window, text="Guardar", command=guardarArchivo, fg="#000000", bg="#E8F6EF", font=("Lato", 12))

inicioScreen()

window.mainloop()

import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)


def neville_method(x, y, e_val):
    m = len(x)
    neville_matrix = np.zeros([m, m], dtype=float)

    for i in range(m):
        neville_matrix[i][0] = y[i]

    for i in range(1, m):
        for j in range(1, m):
            first_multi = (e_val - x[i - j]) * neville_matrix[i][j - 1]
            second_multi = (e_val - x[i]) * neville_matrix[i - 1][j - 1]
            neville_matrix[i][j] = (first_multi - second_multi) / (x[i] - x[i - j])
    return neville_matrix


def newton_fw(x, y):
    m = len(x)

    newton_matrix = np.zeros([m, m], dtype=float)
    for i in range(m):
        newton_matrix[i][0] = y[i]

    for i in range(1, m):
        for j in range(1, m):
            newton_matrix[i][j] = (newton_matrix[i][j - 1] - newton_matrix[i - 1][j - 1]) / (x[i] - x[i - j])

    return newton_matrix


def newton_fw_approx(matrix, x, y, approx):
    p_1 = y[0] + (matrix[1][1] * (approx - x[0]))

    p_2 = p_1 + (matrix[2][2] * ((approx - x[0]) * (approx - x[1])))

    p_3 = p_2 + (matrix[3][3] * ((approx - x[0]) * (approx - x[1]) * (approx - x[2])))

    return p_3


def hermite_calc(x, y, dx):
    m = len(x)
    n = m * 2
    h_matrix = np.zeros([n, n], dtype=float)

    temp = 0
    for i in range(n, 0, -2):
        h_matrix[temp][0] = x[i % 3]
        h_matrix[temp + 1][0] = x[i % 3]
        temp = temp + 2

    temp = 0
    for i in range(n, 0, -2):
        h_matrix[temp][1] = y[i % 3]
        h_matrix[temp + 1][1] = y[i % 3]
        temp = temp + 2

    temp = 0
    while temp < m:
        h_matrix[temp * 2 + 1][2] = dx[temp]
        temp += 1

    for i in range(2, n - 1, 2):
        h_matrix[i][2] = h_calc(h_matrix[i][1], h_matrix[i - 1][1], h_matrix[i][0], h_matrix[i - 1][0])

    for i in range(2, n):
        if i % 2 == 0:
            h_matrix[i][3] = h_calc(h_matrix[i][2], h_matrix[i - 1][2], h_matrix[i][0], h_matrix[i - 1][0])
        if i % 2 == 1:
            h_matrix[i][3] = h_calc(h_matrix[i - 1][2], h_matrix[i][2], h_matrix[i - 2][0], h_matrix[i][0])

    for i in range(3, n):
        if i % 2 == 0:
            h_matrix[i][4] = h_calc(h_matrix[i][2], h_matrix[i - 2][2], h_matrix[i + 1][0], h_matrix[i - 1][0])
        if i % 2 == 1:
            h_matrix[i][4] = h_calc(h_matrix[i - 1][3], h_matrix[i][3], h_matrix[i - 2][0], h_matrix[i][0])

    for i in range(4, n):

        h_matrix[i][5] = h_calc(h_matrix[i][4], h_matrix[i - 1][4], h_matrix[i][0], h_matrix[i - 4][0])

    print(h_matrix, end="\n\n")


def h_calc(tx, ty, dx, dy):
    top = (tx - ty)
    bot = (dx - dy)
    calc = top / bot
    return calc


def cubic_spline(x, y):

    n = len(x)

    matrix = np.zeros([n, n], dtype=float)
    matrix_b = np.zeros([1, n], dtype=float)
    matrix_c = np.zeros([n, 1], dtype=float)

    matrix[0][0] = 1
    matrix[3][3] = 1

    ct = 1

    for i in range(0, n-1, 2):
        matrix[1][i] = x[ct] - x[ct-1]
        ct += 1

    ct = 2

    for i in range(1, n, 2):
        matrix[2][i] = x[ct] - x[ct-1]
        ct += 1

    for i in range(1, n-1):
        for j in range(1, n-1):
            if j == i:
                matrix[i][j] = 2 * (matrix[i][j-1] + matrix[i][j+1])

    print(matrix, end="\n\n")

    for i in range(1, n-1):

        h_n1 = matrix[i][i-1]
        h_n2 = matrix[i][i+1]

        a_n1 = y[i] - y[i-1]
        a_n2 = y[i+1] - y[i]
        matrix_b[0][i] = ((3/h_n2) * a_n2) - ((3 / h_n1) * a_n1)

    print(*matrix_b, end="\n\n")

    for i in range(0, n):
        matrix_c[i] = matrix_b[0][i]

    vector_x_temp = np.linalg.solve(matrix, matrix_c)
    vector_x = np.zeros([1, n], dtype=float)

    for i in range(0, n):
        vector_x[0][i] = vector_x_temp[i]
    print(*vector_x, end="\n\n")


if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=100)
    neville_x = np.asarray([3.600, 3.800, 3.900])
    neville_f = np.asarray([1.675, 1.436, 1.318])

    ans_one = neville_method(neville_x, neville_f, 3.7)

    print(ans_one[2][2], end="\n\n")

    newton_fw_x = np.asarray([7.2, 7.4, 7.5, 7.6])
    newton_fw_f = np.asarray([23.5492, 25.3913, 26.8224, 27.4589])

    ans_two = newton_fw(newton_fw_x, newton_fw_f)

    print("[", end="")
    print(ans_two[1][1], end=", ")
    print(ans_two[2][2], end=", ")
    print(ans_two[3][3], end="]\n\n")

    print(newton_fw_approx(ans_two, newton_fw_x, newton_fw_f, 7.3), end="\n\n")

    hermite_x = np.asarray([3.6, 3.8, 3.9])
    hermite_fx = np.asarray([1.675, 1.436, 1.318])
    hermite_dfx = np.asarray([-1.195, -1.188, -1.182])

    hermite_calc(hermite_x, hermite_fx, hermite_dfx)

    cubic_spline_x = np.asarray([2, 5, 8, 10], dtype=float)
    cubic_spline_fx = np.asarray([3, 5, 7, 9], dtype=float)

    cubic_spline(cubic_spline_x, cubic_spline_fx)



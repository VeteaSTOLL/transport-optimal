import random
import math
import numpy as np

def generate_cloud_ellipse(n, mu, u1, u2):
    points = []
    for _ in range(n):
        r1 = random.random() or 1e-10
        r2 = random.random()
        z1 = math.sqrt(-2 * math.log(r1)) * math.cos(2 * math.pi * r2)
        z2 = math.sqrt(-2 * math.log(r1)) * math.sin(2 * math.pi * r2)
        x = mu[0] + z1 * u1[0] + z2 * u2[0]
        y = mu[1] + z1 * u1[1] + z2 * u2[1]
        points.append((x, y))
    return points

def generate_cloud(n, std):
    return generate_cloud_ellipse(n, (0, 0), (std, 0), (0, std))

def mean_cloud(points):
    sum = [0, 0]
    for p in points:
        sum[0] += p[0]
        sum[1] += p[1]
    sum[0] /= len(points)
    sum[1] /= len(points)
    return tuple(sum)

def variance_cloud(points, mean):
    C2 = [[0,0],[0,0]]
    n = len(points)
    Mu = ((mean[0],), (mean[1],))

    for p in points:
        Xi = ((p[0],), (p[1],))
        S = np.subtract(Xi, Mu)
        ST = np.transpose(S)
        C2 = np.add(C2, np.multiply(S, ST))

    C2 = np.divide(C2, n)
    return C2

def sqrtm(M):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(np.sqrt(w)) @ V.T

def invsqrtm(M):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(1/np.sqrt(w)) @ V.T

def matrix_transport_gauss(Cx, Cy):
    return invsqrtm(Cx) @ sqrtm(sqrtm(Cx) @ Cy @ sqrtm(Cx)) @ invsqrtm(Cx)

def get_slice_direction(X, Y):
    muX = mean_cloud(X)
    muY = mean_cloud(Y)
    Cx = variance_cloud(X, muX)
    Cy = variance_cloud(Y, muY)
    A = matrix_transport_gauss(Cx, Cy)    
    _, vectors  = np.linalg.eigh(A)
    vector = vectors[:, random.randint(0,1)]
    return vector

def sort_indices_with_slice(idx, X, v_dir, start, end):
    idx[start:end] = sorted(idx[start:end], key=lambda i: np.dot(X[i], v_dir))

def _BSP_matching_rec(X, Y, idxX, idxY, start, end, T):
    if end - start == 1:
        T[idxX[start]] = idxY[start]
        return

    sliceX = [X[i] for i in idxX[start:end]]
    sliceY = [Y[j] for j in idxY[start:end]]
    v_dir = get_slice_direction(sliceX, sliceY)

    sort_indices_with_slice(idxX, X, v_dir, start, end)
    sort_indices_with_slice(idxY, Y, v_dir, start, end)

    pivot = (start + end) // 2

    _BSP_matching_rec(X, Y, idxX, idxY, start, pivot, T)
    _BSP_matching_rec(X, Y, idxX, idxY, pivot, end, T)

def BSP_matching(X, Y):
    """
    X, Y : listes de points (non modifiées), len(X) == len(Y)
    Retour : T list[int], T[i] = j signifie X[i] apparié à Y[j].
    """
    assert len(X) == len(Y)
    n = len(X)
    idxX = list(range(n))
    idxY = list(range(n))
    T = [-1] * n
    if n > 0:
        _BSP_matching_rec(X, Y, idxX, idxY, 0, n, T)
    return T

def connected_components(X, Y, T1, T2):
    adj = []


    return []

def bijection_merging(X, Y, T1, T2):
    CC = connected_components(X, Y, T1, T2)

    return
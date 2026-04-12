import random
import math
import numpy as np

def generate_cloud(n, std):
    points = []
    for _ in range(n):
        u1 = random.random() or 1e-10
        u2 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        points.append((z0 * std, z1 * std))
    return points

def mean_cloud(points):
    sum = [0, 0]
    for p in points:
        sum[0] += p[0]
        sum[1] += p[1]
    sum[0] /= len(points)
    sum[1] /= len(points)
    return tuple(sum)

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

import numpy as np
import random

def predict(F, H, Q, P, x=None, B=None, u = None):
    n = F.shape[1]
    if x is None:
        x=np.zeros((n, 1))
    if B or u is None:
        x = x = np.dot(F, x)
    else:
        x = np.dot(F, x) + np.dot(B, u)
    P = np.dot(np.dot(F, P), F.T) + Q
    return x, P


def update(F,H,Q,P,x,R,z):
    n = F.shape[1]
    y = z - np.dot(H, x)
    S = R + np.dot(H, np.dot(P, H.T))
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    x = x + np.dot(K, y)
    I = np.eye(n)
    P = np.dot(np.dot(I - np.dot(K, H), P),
        (I - np.dot(K, H)).T) + np.dot(np.dot(K, R), K.T)
    return x, P

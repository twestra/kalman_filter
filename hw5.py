import numpy as np
import random
from kalmanfilter import predict, update
import matplotlib.pyplot as plt
def example1():
    F = np.array([1]).reshape(1,1)
    H = np.array([1]).reshape(1,1)
    Q = np.array([.0001])
    R = np.array([0.2]).reshape(1, 1)
    n = F.shape[1]
    x = np.array([0]).reshape(1,1)
    P = np.array([1000])
    measurements = []
    predictions = []
    level = []
    for t in range(50):
        t/=10
        level.append(1)
        measurements.append(1+random.uniform(-.3,.3))


    for z in measurements:
        (x,P) = predict(F, H, Q, P, x)
        (x,P) = update(F,H,Q,P,x,R,z)
        print(x)
        predictions.append(np.dot(H, x)[0])


    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.plot(range(len(level)), np.array(level), label = 'Level')
    plt.legend()
    plt.show()
def example2():
    F = np.array([[1,1],[0,1]])
    H = np.array([1, 0]).reshape(1,2)
    Q = np.array([[0, 0], [0, .0001]])
    R = np.array([0.1]).reshape(1, 1)
    n = F.shape[1]
    x = np.zeros((n, 1))
    P = np.array([[1000,0],[0,1000]])
    measurements = []
    predictions = []
    level = []
    for t in range(50):
        t/=10
        level.append(t)
        measurements.append(t+random.uniform(-.3,.3))
    

    for z in measurements:
        (x,P) = predict(F, H, Q, P, x)
        (x, P) = update(F,H,Q,P,x,R,z)
        print(x)
        predictions.append(np.dot(H, x)[0])

    
    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.plot(range(len(level)), np.array(level), label = 'Level')
    plt.legend()
    plt.show()
def example3():
    F = np.array([[1,1],[0,1]])
    H = np.array([1, 0]).reshape(1,2)
    Q = np.array([[0, 0], [0, .8]])
    R = np.array([0.1]).reshape(1, 1)
    n = F.shape[1]
    x = np.zeros((n, 1))
    P = np.array([[1000,0],[0,1000]])
    measurements = []
    predictions = []
    level = []
    
    time = np.arange(0, 50, .1)
    level = np.sin(time)
    
    for t in range(len(level)):
        measurements.append(level[t]+random.uniform(-.3,.3))
    

    for z in measurements:
        (x,P) = predict(F, H, Q, P)
        (x, P) = update(F,H,Q,P,x,R,z)
        print(x)
        predictions.append(np.dot(H, x)[0])

    
    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.plot(range(len(level)), np.array(level), label = 'Level')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    example3()

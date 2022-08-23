"""
author: hova88
data: 2022.08.23
python example of https://www.kalmanfilter.net/multiExamples.html
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque


class KalmanFilter(object):
    def __init__(self, F, P, Q, H, R, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        # State transition matrix 
        self.F = F
        # Estimate covariance matrix
        self.P = P 
        # Process noise matrix
        self.Q = Q 
        # Observation matrix
        self.H = H
        # Measurement covariance matrix
        self.R = R
        
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self):
        # formula 1: predicted (a priori) state estimate
        self.x = self.H @ self.x
        # formula 2: predicted (a priori) estimate covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        # formula 3: Optimal Kalman gain
        K = (self.P @ self.H.T) @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        # formula 4: Updated (a posteriori) state estimate
        self.x = self.x + K @ (z - self.H @ self.x)
        # formula 5: Updated (a posteriori) estimate covariance
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

def constant_acceleration_2d():
    # create a time array from 0~t_stop sampled at 1 second steps
    dt = 1
    t_stop = 100
    t = np.arange(0, t_stop, dt)    
    history_len = 100  # how many trajectory points to display
    lins = np.linspace(1, 100, 100).reshape(-1,1)
    mu_acc , sigma_acc = 0 , 0.2
    mu_vel , sigma_vel = 0 , 0.1
    mu_pos , sigma_pos = 0 , 100
    position_true = np.zeros((100,2))
    acceleration_true = np.ones((100,2))
    velocity_true = np.hstack((lins,lins))
    for i in range(1,len(position_true)):
        position_true[i] = position_true[i-1] + velocity_true[i] * dt + acceleration_true[i] * (dt**2)/2

    # create trajectory
    acceleration_mea = acceleration_true + np.random.normal(mu_acc, sigma_acc, (100,2)) 
    velocity_mea = velocity_true + np.random.normal(mu_vel, sigma_vel, (100,2))
    position_mea = position_true + np.random.normal(mu_pos, sigma_pos, (100,2))
    measurements = np.hstack((position_mea[:,0], velocity_mea[:,0], acceleration_mea[:,0],
                              position_mea[:,1], velocity_mea[:,1], acceleration_mea[:,1])).reshape(6,-1)
    # print(measurements[:,-1])
    F = np.array([[   1,  dt, dt*dt/2, 0.0,  0.0,     0.0], 
                  [ 0.0,   1,      dt, 0.0,  0.0,     0.0], 
                  [ 0.0, 0.0,       1, 0.0,  0.0,     0.0], 
                  [ 0.0, 0.0,     0.0,    1,  dt, dt*dt/2], 
                  [ 0.0, 0.0,     0.0,    0,   1,      dt], 
                  [ 0.0, 0.0,     0.0,    0,   0,       1],])
 
    # P = np.eye(6) * 500 
    P = np.array([[  1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 
                  [  1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 
                  [  1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 
                  [  0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 
                  [  0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 
                  [  0.0, 0.0, 0.0, 1.0, 1.0, 1.0],]) * 500

    Q = np.array([[  (dt**4)/4,  (dt**3)/2, (dt**2)/2,       0.0,       0.0,       0.0], 
                  [  (dt**3)/2,      dt**2,        dt,       0.0,       0.0,       0.0], 
                  [  (dt**2)/2,         dt,         1,       0.0,       0.0,       0.0], 
                  [        0.0,        0.0,       0.0, (dt**4)/4, (dt**3)/2, (dt**2)/2], 
                  [        0.0,        0.0,       0.0, (dt**3)/2,     dt**2,        dt], 
                  [        0.0,        0.0,       0.0, (dt**2)/2,        dt,         1],]) * (sigma_acc**2)

    H = np.eye(6)

    R = np.array([[sigma_pos**2,            0,            0,            0,            0,            0],
                  [           0, sigma_pos**2,            0,            0,            0,            0],
                  [           0,            0, sigma_pos**2,            0,            0,            0],
                  [           0,            0,            0, sigma_pos**2,            0,            0],
                  [           0,            0,            0,            0, sigma_pos**2,            0],
                  [           0,            0,            0,            0,            0, sigma_pos**2]])

    kf = KalmanFilter(F, P, Q, H, R, x0=measurements[:,0:1])

    predictions = np.zeros((6,1))

    for i in range(1,len(measurements[-1])):
        predictions = np.hstack((predictions , H @ kf.predict()))
        kf.update(measurements[:,i:i+1])
    predictions = predictions.T
    
    
    fig = plt.figure(figsize=(11, 14))
    ax = fig.add_subplot(autoscale_on=True, xlim=(-200, 5200), ylim=(-200, 5200))
    ax.set_aspect('equal')
    ax.grid()

    line_mea, = ax.plot([], [], '.-', lw=1)
    trace_mea, = ax.plot([], [], 'o--', lw=1,ms=3)
    history_mea_x, history_mea_y = deque(maxlen=history_len), deque(maxlen=history_len)
    
    trace_true, = ax.plot([], [], '*--', lw=1,ms=3)
    history_true_x, history_true_y = deque(maxlen=history_len), deque(maxlen=history_len)
    
    trace_pred, = ax.plot([], [], 'o-', lw=2,ms=3)
    history_pred_x, history_pred_y = deque(maxlen=history_len), deque(maxlen=history_len)
    
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        # trajectory of measurement
        mea_x = [position_mea[i-1,0], position_mea[i,0]]
        mea_y = [position_mea[i-1,1], position_mea[i,1]]
        if i == 0:
            history_mea_x.clear()
            history_mea_y.clear()
        history_mea_x.appendleft(mea_x[-1])
        history_mea_y.appendleft(mea_y[-1])
        line_mea.set_data(mea_x, mea_y)
        trace_mea.set_data(history_mea_x, history_mea_y)
        
        # trajectory of predictions
        pred_x = [predictions[i-1,0], predictions[i,0]]
        pred_y = [predictions[i-1,3], predictions[i,3]]
        if i == 0:
            history_pred_x.clear()
            history_pred_y.clear()
        history_pred_x.appendleft(pred_x[-1])
        history_pred_y.appendleft(pred_y[-1])
        trace_pred.set_data(history_pred_x, history_pred_y)
        
        # trajectory of ground true
        true_x = [position_true[i-1,0], position_true[i,0]]
        true_y = [position_true[i-1,1], position_true[i,1]]
        history_true_x.appendleft(true_x[-1])
        history_true_y.appendleft(true_y[-1])
        trace_true.set_data(history_true_x, history_true_y)


        time_text.set_text(time_template % (i))
        return line_mea, trace_mea, trace_true, trace_pred, time_text


    ani = animation.FuncAnimation(
        fig, animate, 99, interval=dt*100, blit=True)
    plt.show()
    
if __name__ == '__main__':
    constant_acceleration_2d()
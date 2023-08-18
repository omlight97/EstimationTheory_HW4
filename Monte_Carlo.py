import numpy as np

import numpy as np

import matplotlib.pyplot as plt

from Kalman_Filter import KF_1D_kinematics as KF

def Monte_Carlo_test(N:int) -> list:

    time = 100
    location_error_mat = np.zeros((N,time))
    vel_error_mat = np.zeros((N,time))
    location_var_mat = np.zeros((N,time))
    vel_var_mat = np.zeros((N,time))

    for iteration in range(N):

        
        dt = 0.1
        x_sys = 0
        v_sys = 1
        acceleration = 4
        system_variance = 0.5**2
        measurement_variance = 4**2

        covariance = []
        state = []
        real_x = []
        real_vx = []
        state_error = []
        innovation = []

        measure_timing = 20

        kf = KF(initial_x=x_sys,initial_vx=v_sys,system_var=system_variance,accel=acceleration)

        for t in range(time):
            
            #Data creation   
            real_x.append(x_sys)
            real_vx.append(v_sys)
            covariance.append(kf.covariance)
            state.append(kf.mean)
            state_error.append(np.array([x_sys,v_sys]).reshape(2,1) - kf.mean)
            kf.predict(dt=dt)

            if t!=0 and t % measure_timing == 0:
                kf.update(measurement=x_sys+np.random.rand()*np.sqrt(measurement_variance)  ,
                meas_var=measurement_variance)
                innovation.append(kf.innovation)
        
            x_sys =  x_sys + v_sys * dt + 0.5 * acceleration * dt ** 2
            v_sys = v_sys + acceleration * dt

        location_error_mat[iteration,:] = [err_tmp[0][0] for err_tmp in state_error]
        vel_error_mat[iteration,:] = [err_tmp[1][0] for err_tmp in state_error]    
    return location_error_mat,vel_error_mat


err_location_data, err_velocity_data, location_var_mat, vel_var_mat= Monte_Carlo_test(5000)

plt.figure()
plt.subplot(2,1,1)
plt.plot(err_location_data.T)
plt.subplot(2,1,2)
plt.plot(err_velocity_data.T)

plt.show()
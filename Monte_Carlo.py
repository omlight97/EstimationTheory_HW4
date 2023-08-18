import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from Kalman_Filter import KF_1D_kinematics as KF


def Monte_Carlo_test(N:int) -> list:

    time = 100
    location_error_mat = np.zeros((N,time))
    vel_error_mat = np.zeros((N,time))

    for iteration in range(N):

        
        dt = 0.1
        x_sys = 0
        v_sys = 1
        acceleration = 4
        system_variance = 3**2
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
                kf.update(measurement= x_sys + np.random.rand() * np.sqrt(measurement_variance)  ,
                meas_var=measurement_variance)
                innovation.append(kf.innovation)
        
            x_sys =  x_sys + v_sys * dt + 0.5 * acceleration * dt ** 2
            v_sys = v_sys + acceleration * dt

        location_error_mat[iteration,:] = [err_tmp[0][0] for err_tmp in state_error]
        vel_error_mat[iteration,:] = [err_tmp[1][0] for err_tmp in state_error]

    location_variance = [np.std(i) for i in location_error_mat.T]
    vel_variance = [np.std(i) for i in vel_error_mat.T]
    cov_location_kf = [np.sqrt(i[0,0]) for i in covariance]
    cov_vel_kf = [np.sqrt(i[1,1]) for i in covariance]
    state_location_kf = [i[0] for i in state_error]
    state_vel_kf = [i[1] for i in state_error]

        
    return location_error_mat,vel_error_mat, location_variance, vel_variance, cov_location_kf, cov_vel_kf, state_location_kf , state_vel_kf


err_location_data, err_velocity_data, loc_var, vel_var, covariance_loc, covariance_vel, kf_error_state, kf_error_vel = Monte_Carlo_test(5000)

#Plotting 5000 simulations


plt.figure()
plt.suptitle(r'Mote Carlo (5000 Simulations) - $\tilde{x}$')
plt.subplot(2,1,1)
plt.title('Location')
plt.plot(err_location_data.T)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel(r'$\tilde{x}$ Location [m]', fontsize=12)

plt.subplot(2,1,2)
plt.title('Velocity')
plt.plot(err_velocity_data.T)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel(r'$\tilde{x}$ Velocity [m/s]', fontsize=12)

plt.figure()
plt.suptitle(r'$\tilde{x}$ - Monte Carlo Vs. KF')
plt.subplot(2,1,1)
plt.plot(np.mean(err_location_data.T,axis=1))
plt.plot(kf_error_state)
plt.plot(np.mean(err_location_data.T ,axis=1) + np.sqrt(loc_var), "--r")
plt.plot(np.mean(err_location_data.T,axis=1) - np.sqrt(loc_var), "--r")
plt.title('Location')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel(r'$\tilde{x}$ Location [m]', fontsize=12)
plt.legend(['Location - Monte Carlo','Location - KF','Location + STD (MC)'])


plt.subplot(2,1,2)
plt.plot(np.mean(err_velocity_data.T,axis=1))
plt.plot(kf_error_vel)
plt.plot(np.mean(err_velocity_data.T ,axis=1) + np.sqrt(vel_var), "--r")
plt.plot(np.mean(err_velocity_data.T,axis=1) - np.sqrt(vel_var), "--r")
plt.title('Velocity')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel(r'$\tilde{x}$ Velocity [m/s]', fontsize=12)
plt.legend(['Velocity - Monte Carlo','Velocity - KF','Velocity + STD (MC)'])

plt.figure()
plt.plot(np.sqrt(loc_var), "--r")
plt.plot(- np.sqrt(loc_var), "--r")
plt.plot([np.sqrt(i) for i in covariance_vel], "--b")
plt.plot([-np.sqrt(i) for i in covariance_vel], "--b")
plt.title(r'$\sigma$ - Monte Carlo Vs. KF (Location)')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('STD', fontsize=12)

plt.show()
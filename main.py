import numpy as np
import matplotlib.pyplot as plt
from Kalman_Filter import KF_1D_kinematics as KF
from scipy.stats import rv_discrete

different_ic_flag = True
accel_change_flag = False


time = 100
dt = 0.1
time_vec = np.arange(0,time+dt,dt)
x_sys = 0
v_sys = 1
x_zero_different = 5
v_zero_different = 10
acceleration = 0.2
system_variance = 0.5**2
measurement_variance = 5**2
measure_timing = 20

delay = 0 #constant

#random delay
# values_of_delay = [i for i in np.linspace(1,8,8)] 
# delay = rv_discrete(name = 'uniform',values = (values_of_delay,[1 / len(values_of_delay) for _ in range(len(values_of_delay))]))

covariance = []
state = []
real_x = []
real_vx = []
state_error = []
innovation = []


# Testing KF with initial conditions different from the actual target's IC
if different_ic_flag:
    kf = KF(initial_x = x_zero_different,
            initial_vx = v_zero_different,
            system_var = system_variance,
            accel = acceleration)
else:
    kf = KF(initial_x = x_sys,
            initial_vx = v_sys,
            system_var = system_variance,
            accel = acceleration)

#Data creation
for t in time_vec:

    #changing input midcourse - optional
    if accel_change_flag and int(t) == 50:
       acceleration = - 0.2
 
    real_x.append(x_sys)
    real_vx.append(v_sys)
    covariance.append(kf.covariance)
    state.append(kf.mean)
    state_error.append(np.array([x_sys,v_sys]).reshape(2,1) - kf.mean)
    kf.predict(dt=dt)
    

    if int(t) != 0 and t % measure_timing == 0 and int(t) != time and not(delay):
        kf.update(measurement = x_sys + np.random.rand() * np.sqrt(measurement_variance)  ,
        meas_var = measurement_variance)
        innovation.append(kf.innovation)
    elif int(t) != 0 and t % measure_timing == 0 and int(t) != time:
        # delta = delay.rvs() #random
        delta = delay #constant
        kf.delayed_update(measurement = x_sys + np.random.rand() * np.sqrt(measurement_variance)  ,
        meas_var = measurement_variance, delay = delta, dt=dt)
        innovation.append(kf.dinnovation)
        print(f"update at time = {int(t)} seconds was made with a delay of {delta} seconds")

    acceleration_with_noise = acceleration + np.random.normal(scale = system_variance)
    x_sys =  x_sys + v_sys * dt + 0.5 * acceleration_with_noise * dt ** 2
    v_sys = v_sys + acceleration_with_noise * dt

#plotting
plt.figure()

plt.subplot(2,1,1)
plt.title("X(t)")
plt.plot(time_vec,[x_tmp[0] for x_tmp in state], "b")
plt.plot(time_vec,real_x, "g")
plt.plot(time_vec,[x_tmp[0] +  np.sqrt(cov_tmp[0, 0]) for x_tmp, cov_tmp in zip(state, covariance)], 'r--')
plt.plot(time_vec,[x_tmp[0] -  np.sqrt(cov_tmp[0, 0]) for x_tmp, cov_tmp in zip(state, covariance)], 'r--')
plt.xlabel('Time [s]', fontsize = 12)
plt.ylabel('X [m]', fontsize = 12)
plt.grid(True)
plt.legend(['Location from KF','Real Location','Estimator + STD'])

plt.subplot(2,1,2)
plt.title("Vx(t)")
plt.plot(time_vec,[v_tmp[1] for v_tmp in state], "b")
plt.plot(time_vec,real_vx, "g")
plt.plot(time_vec,[v_tmp[1] +  np.sqrt(cov_tmp[1, 1]) for v_tmp, cov_tmp in zip(state, covariance)], 'r--')
plt.plot(time_vec,[v_tmp[1] -  np.sqrt(cov_tmp[1, 1]) for v_tmp, cov_tmp in zip(state, covariance)], 'r--')
plt.xlabel('Time [s]', fontsize = 12)
plt.ylabel('Vx [m/s]', fontsize = 12)
plt.grid(True)
plt.legend(['Velocity from KF','Real Velocity','Estimator + STD'])

plt.figure()

plt.subplot(2,1,1)
plt.grid(True)
plt.title(r'system error - Location $\tilde{x}$')
plt.plot(time_vec,[i[0][0] for i in state_error], "b")
plt.plot(time_vec,[err_tmp[0] + np.sqrt(cov_tmp[0, 0]) for err_tmp,cov_tmp in zip(state_error,covariance)], 'r--')
plt.plot(time_vec,[err_tmp[0] - np.sqrt(cov_tmp[0, 0]) for err_tmp,cov_tmp in zip(state_error,covariance)], 'r--')
plt.xlabel('Time [s]', fontsize = 12)
plt.ylabel(r'$\tilde{x}$ Location [m]', fontsize = 12)
plt.legend(['Location','STD'])


plt.subplot(2,1,2)
plt.grid(True)
plt.title(r'system error - Velocity $\tilde{x}$')
plt.plot(time_vec,[i[1][0] for i in state_error], "b")
plt.plot(time_vec,[err_tmp[1] + np.sqrt(cov_tmp[1, 1]) for err_tmp,cov_tmp in zip(state_error,covariance)], 'r--')
plt.plot(time_vec,[err_tmp[1] - np.sqrt(cov_tmp[1, 1]) for err_tmp,cov_tmp in zip(state_error,covariance)], 'r--')
plt.xlabel('Time [s]', fontsize = 12)
plt.ylabel(r'$\tilde{x}$ Velocity [m/s]', fontsize = 12)
plt.legend(['Velocity','STD'])

plt.figure()
plt.grid(True)
plt.title(r'Innovation $\tilde{z}$')
plt.scatter(list([np.arange(measure_timing,time,measure_timing)][0]),[i[0][0] for i in innovation])
plt.xlabel('Time [s]', fontsize = 12)
plt.ylabel(r'$\tilde{z} [m]$', fontsize = 12)

plt.show()
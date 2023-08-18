import numpy as np

import matplotlib.pyplot as plt

from Kalman_Filter import KF_1D_kinematics as KF

time = 100
dt = 0.1

x_sys = 0
v_sys = 1
different_ic_flag = False
accel_change_flag = False
x0 = 5
v0 = 10
acceleration = 4
system_variance = 0.5**2
measurement_variance = 0.1**2

covariance = []
state = []
real_x = []
real_vx = []
state_error = []
innovation = []

measure_timing = 20


if different_ic_flag:
    kf = KF(initial_x=x0,initial_vx=v0,system_var=system_variance,accel=acceleration)
else:
    kf = KF(initial_x=x_sys,initial_vx=v_sys,system_var=system_variance,accel=acceleration)

for t in range(time):
    if accel_change_flag and t >= 50:
       kf = KF(initial_x=x0,initial_vx=v0,system_var=system_variance,accel=4-6*(t/time))
       acceleration = 4-4*(t/time)

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


plt.figure()

plt.subplot(2,1,1)
plt.title("X(t)")

#plotting
plt.plot([x_tmp[0] for x_tmp in state], "b")
plt.plot(real_x, "g")
plt.plot([x_tmp[0] +  np.sqrt(cov_tmp[0, 0]) for x_tmp, cov_tmp in zip(state, covariance)], 'r--')
plt.plot([x_tmp[0] -  np.sqrt(cov_tmp[0, 0]) for x_tmp, cov_tmp in zip(state, covariance)], 'r--')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('X [m]', fontsize=12)
plt.legend(['Location from KF','Real Location','Estimator + STD'])

plt.subplot(2,1,2)
plt.title("Vx(t)")
plt.plot([v_tmp[1] for v_tmp in state], "b")
plt.plot(real_vx, "g")
plt.plot([v_tmp[1] +  np.sqrt(cov_tmp[1, 1]) for v_tmp, cov_tmp in zip(state, covariance)], 'r--')
plt.plot([v_tmp[1] -  np.sqrt(cov_tmp[1, 1]) for v_tmp, cov_tmp in zip(state, covariance)], 'r--')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Vx [m/s]', fontsize=12)
plt.legend(['Velocity from KF','Real Velocity','Estimator + STD'])

plt.figure()
plt.subplot(2,1,1)

plt.title(r'system error - Location $\tilde{x}$')
plt.plot([i[0][0] for i in state_error], "b")
plt.plot([err_tmp[0] + np.sqrt(cov_tmp[0, 0]) for err_tmp,cov_tmp in zip(state_error,covariance)], 'r--')
plt.plot([err_tmp[0] - np.sqrt(cov_tmp[0, 0]) for err_tmp,cov_tmp in zip(state_error,covariance)], 'r--')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel(r'$\tilde{x}$ Location [m]', fontsize=12)
plt.legend(['Location','STD'])


plt.subplot(2,1,2)

plt.title(r'system error - Velocity $\tilde{x}$')
plt.plot([i[1][0] for i in state_error], "b")
plt.plot([err_tmp[1] + np.sqrt(cov_tmp[1, 1]) for err_tmp,cov_tmp in zip(state_error,covariance)], 'r--')
plt.plot([err_tmp[1] - np.sqrt(cov_tmp[1, 1]) for err_tmp,cov_tmp in zip(state_error,covariance)], 'r--')
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel(r'$\tilde{x}$ Velocity [m/s]', fontsize=12)
plt.legend(['Velocity','STD'])

plt.figure()

plt.title(r'Innovation $\tilde{z}$')
plt.scatter(list([np.arange(measure_timing,time,measure_timing)][0]),[i[0][0] for i in innovation])
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel(r'$\tilde{z}$', fontsize=12)

plt.show()
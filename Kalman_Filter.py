import numpy as np

class KF_1D_kinematics:
    """uses standard Kalman Filter to estimate state of 1D movement"""

    def __init__(self, initial_x: float, initial_vx: float,
                       system_var: float, accel: float) -> None:
        """Args:
            initial_x(float): initial location of target at t=0
            initial_vx(float): initial velocity of target at t=0
            system_var(float): variance of the system's white gaussian noise
            accel(float): acceleration of the system - a known input
        """
        self.state = np.array([[initial_x],[initial_vx]])
        self.P = np.eye(2) #estimation error's covariance. Initial guess
        self.accel = accel
        self.system_var = system_var
        self._innovation = []
        self._delta = 0
        self._delayed_state = []
        self._delayed_P = []
        self._time_passed = 0
        self._timer = False

    def predict(self, dt: float) -> None:
        """
        Predicts the state of the system according to KF model
         Args:
            dt(float): advancement in time - predict state at t+dt 
        """
        Phi = np.array([[1,dt],[0,1]])
        Gamma = np.array([0.5*dt**2,dt]).reshape(2,1)
        Psi = np.array([1,1]).reshape(2,1)

        predicted_state = Phi.dot(self.state)+Gamma.dot(self.accel)
        predicted_P = Phi.dot(self.P).dot(Phi.T)+Psi.dot(self.system_var).dot(Psi.T)

        if self._timer and int(self._time_passed) - self._delta == 0:
            self.state= self._delayed_state
            self.P = self._delayed_P
            self._delta = 0
            self._time_passed = 0
            self._timer = False
        elif self._timer and int(self._time_passed) - self._delta != 0:
            self._time_passed = self._time_passed + dt
            self.state= predicted_state
            self.P = predicted_P
        else:
            self.state= predicted_state
            self.P = predicted_P

    def update(self,measurement: float ,meas_var: float) -> None:
        """
        Updates the state of the system according to KF model, using a measurement
         Args:
            measurement(float): measurement of the location at specific time 
            meas_var(float): variance of the virtual sensors' white gaussian noise
        """
        H = np.array([1,0]).reshape((1,2))
        z = np.array([measurement])

        self._innovation = z - H.dot(self.state)

        R = np.array(meas_var)

        S = H.dot(self.P).dot(H.T) + R

        K_gain = self.P.dot(H.T).dot(np.linalg.inv(S))
        
        y = np.eye(2) - K_gain.dot(H)

        updated_state = self.state + K_gain.dot((z - H.dot(self.state)))

        updated_P = y.dot(self.P).dot(y.T) + K_gain.dot(R).dot(K_gain.T)
        
        self.state = updated_state
        self.P = updated_P

    def delayed_update(self,measurement: float ,meas_var: float, delay: float, dt:float) -> None:
        """
        Updates the state of the system according to KF model, using a measurement
         Args:
            measurement(float): measurement of the location at specific time 
            # meas_var(float): variance of the virtual sensors' white gaussian noise
        """
        self._delta = delay

        H = np.array([1,0]).reshape((1,2))

        z = np.array([measurement])

        self._dinnovation = z - H.dot(self.state)
        
        R = np.array(meas_var)

        S = H.dot(self.P).dot(H.T) + R

        K_gain = self.P.dot(H.T).dot(np.linalg.inv(S))
        
        y = np.eye(2) - K_gain.dot(H)

        updated_state = self.state + K_gain.dot((z - H.dot(self.state)))

        updated_P = y.dot(self.P).dot(y.T) + K_gain.dot(R).dot(K_gain.T)
        
        self._delayed_state = updated_state
        self._delayed_P = updated_P

        KF_1D_kinematics.predict(self, dt=dt)
        self._timer = True
        

    @property
    def covariance(self) -> np.array:
        return self.P

    @property
    def mean(self) -> np.array:
        return self.state
    
    @property
    def innovation(self) -> np.array:
        return self._innovation
    
    @property
    def dinnovation(self) -> np.array:
        return self._dinnovation
        


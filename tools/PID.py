class PID:
    """PID Controller"""

    def __init__(self, Kp, Ki=0.0, Kd=0.0, dt=0.01):
        # Set input parameters
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.dt = dt

        # Define integration and derivative methods
        self._I = lambda error: self.I + self.Ki * error * self.dt
        self._D = lambda error: self.Kd * (error - self.last_error) / self.dt

        # Set initial PID values
        self.P, self.I, self.D = 0.0, 0.0, 0.0
        self.last_error = 0.0
        self.last_time = 0.0
        self.count = 0

    def __call__(self, error, current_time):
        """Calculates PID value for given reference feedback

        .. math::
            output(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}

        """

        elapsed_time = round((current_time - self.last_time)/0.00001)*0.00001

        if elapsed_time >= self.dt:
            # Proportional Term
            self.P = self.Kp * error

            # Integral Term
            self.I = self._I(error)

            # Derivative Term
            self.D = self._D(error)
            self.last_error = error
            self.last_time = current_time
            self.count += 1

        return self.P + self.I + self.D

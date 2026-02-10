import numpy as np

class WirelessModel:
    def __init__(self):
        # realistic constants
        self.B = 10e6          # 10 MHz bandwidth
        self.N0 = 1e-13       # noise power
        self.alpha = 3.5      # pathloss exponent

        # powers
        self.P_device = 0.1   # 100 mW
        self.P_edge = 1.0     # 1 W

    def rate(self, P, distance):
        h = 1.0 / (distance ** self.alpha + 1e-9)
        snr = P * h / self.N0
        return self.B * np.log2(1 + snr)

    def energy(self, bits, distance, is_edge=False):
        P = self.P_edge if is_edge else self.P_device
        R = self.rate(P, distance)
        delay = bits / R
        return P * delay

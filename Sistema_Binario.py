from Globals import *
import numpy as np
import matplotlib.pyplot as plt


class BinarySystem(object):
    def __init__(self, m1, m2, T, ecc):
        self.m1 = m1
        self.m2 = m2
        self.T = T
        self.ecc = ecc
        self.dtout = 0.1 * YR
        self.tfin = 2 * 10.98 * YR
        self.nout = int(self.tfin/self.dtout)
        self.time = np.linspace(0, self.tfin, self.nout)
        self.tol = 1e-10
        self.G = 6.67259e-8 # kg^-1 s^-2
        self.M = self.m1 + self.m2
        self.Rx = 30 * AU
        self.Ry = 30 * AU
            
    def reduced_mass(self):
        """
        Calculates the reduced mass of a binary system
        
        Attributes
        ----------
        self.m1: float
            mass of the first star
        self.m2: float
            mass of the second star
        Returns
        -------
        float: Reduced mass of the system as a float
        """
        return (self.m1*self.m2)/(self.m1 + self.m2)
        
    def semi_major_axis(self):
        """
        Calculate the semi-major axis of the binary system using the orbital
        period and the total mass of the system.
        
        Attributes
        ----------
        self.T: float
            Orbital period of the binary system 
        self.G: float
            Gravitational constant
        self.M: float
            Total mass of the binary system.
        
        Returns
        -------
        float: The semi-major axis of the binary system.
        """
        return ((self.T**2)*self.G*(self.M))**(1/3)
    
    def center_of_mass(self):
        """
        This function calculates the cenger of mass of the binary system
        
        Parameters
        ----------
        self: Object
            The object of the BinarySystem class

        Returns
        -------
            float: The center of mass of the binary system  
        """
        return (self.m1 * self.m1)/(self.m1+self.m2)
    
    def kepler_problem(self, phase, velocities=False):
        """
        Compute the positions ad velocities of two objects in a binary
        system at a given phase

        Parameters
        ----------
        self: Object
            The object of the BinarySystem class
        phase: float
            the phase of the binary system, in the range [0, 1]
        
        Returns
        -------
            The x and y positions and velocities of the two objects 
            in the binary system 
        """
        a = (self.T**2*self.G*(self.m1+self.m2)/(4*np.pi**2))**(1/3)
        l = self.reduced_mass()*(1-self.ecc)*a*np.sqrt((1+self.ecc)/(1-self.ecc)*self.G*(self.m1+self.m2)/a)
        average_anomaly = 2*np.pi*phase
        
        if average_anomaly == 0:
            E = 0
        if self.ecc > 0.8:
            x = np.copysign(np.pi, average_anomaly)
        else:
            x = average_anomaly
        rel_err = 2*self.tol
        
        while rel_err > self.tol:
            xn = x - (x-self.ecc*np.sin(x)-average_anomaly)/(1-self.ecc*np.cos(x))
            if x != 0:
                rel_err = abs((xn-x)/x)
                x = xn
        E = x
        
        theta = 2.0*np.arctan(np.sqrt((1+self.ecc)/(1-self.ecc))*np.tan(E/2))
        rad = a*(1-self.ecc**2)/(1+self.ecc*np.cos(theta))
        
        thetadot = l/(self.reduced_mass()*rad**2)
        raddot = a*self.ecc*np.sin(theta)*(1-self.ecc**2)/(1+self.ecc*np.cos(theta))**2 * thetadot
        
        radx = rad*np.cos(theta)
        rady = rad*np.sin(theta)
        
        x1 = self.Rx - self.m2/(self.m1+self.m2)*radx
        y1 = self.Ry - self.m2/(self.m1+self.m2)*rady
        x2 = self.Rx + self.m1/(self.m1+self.m2)*radx
        y2 = self.Ry + self.m1/(self.m1+self.m2)*rady
        
        if velocities == True:
            vx = raddot*np.cos(theta)-rad*thetadot*np.sin(theta)
            vy = raddot*np.sin(theta)+rad*thetadot*np.cos(theta)
            vx1 = -self.m2/(self.m1 + self.m2)*vx
            vy1 = -self.m2/(self.m1 + self.m2)*vy
            vx2 = self.m1/(self.m1 + self.m2)*vx
            vy2 = self.m1/(self.m1 + self.m2)*vy
            return x1/AU, y1/AU, x2/AU, y2/AU, vx1, vy1, vx2, vy2
        else:
            return x1/AU, y1/AU, x2/AU, y2/AU
        
    def trajectories(self):
        """
        Calculate the star trajectories of the binary system 
        
        Parameters
        ----------
        self: Object
            The object of the BinarySystem class
        Returns: tuple
            pos_star1_x: List of x position in the time of the first star
            pos_star1_y: List of y position in the time of the first star
            pos_star2_x: List of x position in the time of the second star
            pos_star2_y: List of 7 position in the time of the second star
        """
        pos_star1_x = []; pos_star1_y = []
        pos_star2_x = []; pos_star2_y = []
        time_len = len(self.time)
        for i in range(time_len):
            phase = ((self.time[i]*t_sc)%self.T)/self.T + 0.25
            r = self.kepler_problem(phase)
            pos_star1_x.append(r[0])
            pos_star1_y.append(r[1])
            pos_star2_x.append(r[2])
            pos_star2_y.append(r[3])
        return pos_star1_x, pos_star1_y, pos_star2_x, pos_star2_y
    

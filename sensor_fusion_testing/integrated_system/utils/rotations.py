import numpy as np

def skew_symmetric(v):
    """Convert a 3x1 vector to a skew symmetric matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def angle_normalize(a):
    """Normalize angle to [-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))

def omega(w, dt):
    """Compute the quaternion update matrix."""
    wx, wy, wz = w
    return np.array([
        [1, -wx*dt/2, -wy*dt/2, -wz*dt/2],
        [wx*dt/2, 1, wz*dt/2, -wy*dt/2],
        [wy*dt/2, -wz*dt/2, 1, wx*dt/2],
        [wz*dt/2, wy*dt/2, -wx*dt/2, 1]
    ])

class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0, axis_angle=None):
        if axis_angle is not None:
            # Convert axis-angle to quaternion
            angle = np.linalg.norm(axis_angle)
            if angle > 0:
                axis = axis_angle / angle
                self.w = np.cos(angle/2)
                self.x = axis[0] * np.sin(angle/2)
                self.y = axis[1] * np.sin(angle/2)
                self.z = axis[2] * np.sin(angle/2)
            else:
                self.w = 1.0
                self.x = 0.0
                self.y = 0.0
                self.z = 0.0
        else:
            self.w = w
            self.x = x
            self.y = y
            self.z = z
            
    def to_numpy(self):
        """Convert to numpy array."""
        return np.array([self.w, self.x, self.y, self.z])
        
    def to_mat(self):
        """Convert to rotation matrix."""
        return np.array([
            [1 - 2*self.y**2 - 2*self.z**2, 2*self.x*self.y - 2*self.w*self.z, 2*self.x*self.z + 2*self.w*self.y],
            [2*self.x*self.y + 2*self.w*self.z, 1 - 2*self.x**2 - 2*self.z**2, 2*self.y*self.z - 2*self.w*self.x],
            [2*self.x*self.z - 2*self.w*self.y, 2*self.y*self.z + 2*self.w*self.x, 1 - 2*self.x**2 - 2*self.y**2]
        ])
        
    def quat_mult_left(self, q):
        """Multiply quaternion from the left."""
        w = self.w * q[0] - self.x * q[1] - self.y * q[2] - self.z * q[3]
        x = self.w * q[1] + self.x * q[0] + self.y * q[3] - self.z * q[2]
        y = self.w * q[2] - self.x * q[3] + self.y * q[0] + self.z * q[1]
        z = self.w * q[3] + self.x * q[2] - self.y * q[1] + self.z * q[0]
        return np.array([w, x, y, z]) 
import numpy as np




class HeavyTails():
    def __init__(self, d = 2,  Sigma = None, tau = None, s = None, mode = "diana1"):

        if Sigma is not None: 
            self.Sigma = Sigma
            self.dim = self.Sigma.shape[-1]
        
        else:
            self.dim = d
            self.Sigma = np.eye(self.dim)
        
        if mode == "diana1":
            self.Sigma = np.array([[1.,0.4], [0.4,1.]])
            self.dim = 2
            self.tau = 1/np.array([1.2,1.2])
            self.s = -np.array([0.2, 0.2])
        
        elif mode == "diana2":
            self.Sigma = np.array([[1.,0.4], [0.4,1.]])
            self.dim = 2
            self.tau = 1/np.array([1.2,1.2])
            self.s = -np.array([0.2, 0.5])
        
        elif mode == "diana3":
            self.Sigma = np.array([[1.,0.4], [0.4,1.]])
            self.dim = 2
            self.tau = 1/np.array([1.2,1.4])
            self.s = -np.array([0.2, 0.2])

        
    def prob(self, z):
        z0, b, d = self.sas_elliptical(z)
        d_logdet = np.linalg.slogdet(self.Sigma)[1]
        invS = np.linalg.inv(self.Sigma)
        
        # quadratic form
        # flatten samples into rows if needed
        z0_flat = z0.reshape(-1, z0.shape[-1])
        qf = np.einsum("ni,ij,nj->n", z0_flat, invS, z0_flat)
        
        # Jacobian factor per sample
        jac = np.prod(d.reshape(-1, d.shape[-1]), axis=1)
        
        # multivariate normal density at z0
        d_dim = z0.shape[-1]
        norm_const = (2*np.pi)**(-0.5*d_dim) * np.exp(-0.5*d_logdet)
        dens_flat = norm_const * np.exp(-0.5*qf) * jac
        
        return dens_flat.reshape(z0.shape[:-1])
    

        
    def log_prob(self, z):    
        return np.log(self.prob(z))

    def gradient_log_density(self, z):
        """
        Returns array of same shape as z, giving ∇_z log f(z).
        """
        z = np.asarray(z)
        z0, b, d = self.sas_elliptical(z)
        invS = np.linalg.inv(self.Sigma)
        
        # compute m = Σ^{-1} z0 for each sample
        # if multiple samples, flatten
        flat_z0 = z0.reshape(-1, z0.shape[-1])
        m_flat = flat_z0.dot(invS.T)       # shape (n, d)
        m = m_flat.reshape(z0.shape)       # back to (..., d)
        
        # elementwise terms
        inv_tau = 1.0 / self.tau
        sqrt1pz2 = np.sqrt(1 + z*z)
        term1 = - d * m                      # shape (..., d)
        term2 = np.tanh(b) * inv_tau / sqrt1pz2
        term3 = - z / (1 + z*z)
        
        return term1 + term2 + term3

    def compute_KL(self, vgmm, noise = None, component_indices = None , B = 1000):
        samples = vgmm.sample(B, noise, component_indices)

        return (vgmm.log_prob(samples[:,None]) - self.log_prob(samples)).mean()

    def sas_elliptical(self, z):
        z = np.asarray(z)
        inv_tau = 1.0 / self.tau
        a = np.arcsinh(z)                     # shape (..., d)
        b = inv_tau * (a + self.s)                 # shape (..., d)
        z0 = np.sinh(b)                       # shape (..., d)
        d = np.cosh(b) * inv_tau / np.sqrt(1 + z*z)
        return z0, b, d


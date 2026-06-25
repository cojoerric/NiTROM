import numpy as np
import scipy.linalg as sciplin


class full_order_model:
    def __init__(self, A2, A3, B, C):

        self.A2 = A2  # Linear tensor (i.e., a matrix)
        self.A3 = A3  # Quadratic tesors (i.e., third-order)
        self.B = B  # The input matrix
        self.C = C  # The output matrix

    def evaluate_fom_dynamics(self, t, q, f):
        """
        Evaluate the FOM dynamics: dx/dt = Ax + H:xx^T + fu(t),
        where A is a second-order tensor, H a third-order tensor and
        fu is an scipy interpolator for the input
        """
        return self.A2 @ q + np.einsum("ijk,j,k", self.A3, q, q) + f(t)

    def compute_output(self, q):
        return self.C @ q

    def compute_output_derivative(self, q):
        return self.C

    def assemble_petrov_galerkin_tensors(self, Phi, Psi):
        """
        Compute the Petrov-Galerkin projection of the tensors
        A2, A3 and B (which define the FOM dynamics)
        """

        n, r = Phi.shape
        PhiF = Phi @ sciplin.inv(Psi.T @ Phi)

        A2r = Psi.T @ self.A2 @ PhiF
        A3r = np.zeros((r, r, r))
        for i in range(r):
            for j in range(r):
                A3r[:, i, j] = Psi.T @ np.einsum(
                    "kij,i,j", self.A3, PhiF[:, i], PhiF[:, j]
                )

        Br = Psi.T @ self.B
        Cr = self.compute_output(PhiF)

        return (A2r, A3r), (Br, Cr)

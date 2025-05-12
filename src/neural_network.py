import autograd.numpy as np
from autograd import jacobian
from einops import rearrange

class MultiLayerNeuralNetwork:
    def __init__(self, data_dim, hidden_units, n_layers, output_dim):
        """
        data_dim    : D, input dimension
        hidden_units: h, # ReLU units per hidden layer
        n_layers    : L, # hidden layers
        output_dim  : C, # output dimension
        """
        self.data_dim   = data_dim
        self.h          = hidden_units
        self.L          = n_layers
        self.C          = output_dim

        # precompute how to slice the flat vector into weights/biases
        self.param_slices = []
        offset = 0

        # Layer 1: W1 (h×D), b1 (h)
        self.param_slices.append(("W", offset, offset + self.h * self.data_dim))
        offset += self.h * self.data_dim
        self.param_slices.append(("b", offset, offset + self.h))
        offset += self.h

        # Layers 2…L: each Wℓ (h×h), bℓ (h)
        for _ in range(self.L - 1):
            self.param_slices.append(("W", offset, offset + self.h * self.h))
            offset += self.h * self.h
            self.param_slices.append(("b", offset, offset + self.h))
            offset += self.h

        # Final layer: V (C×h), c (C)
        self.param_slices.append(("W", offset, offset + self.C * self.h))
        offset += self.C * self.h
        self.param_slices.append(("b", offset, offset + self.C))
        offset += self.C

        self.param_dim = offset

    def unpack_params(self, flat):
        """
        flat: (..., param_dim)
        returns: list Ws, list bs
        Ws[l].shape = (..., rows, cols)
        bs[l].shape = (..., rows)
        """
        Ws, bs = [], []
        for tag, lo, hi in self.param_slices:
            chunk = flat[..., lo:hi]
            if tag == "W":
                # pick the right shape in the order we built them
                if len(Ws) == 0:
                    rows, cols = self.h, self.data_dim
                elif len(Ws) < self.L:
                    rows, cols = self.h, self.h
                else:
                    rows, cols = self.C, self.h
                Ws.append(chunk.reshape(*chunk.shape[:-1], rows, cols))
            else:  # tag == "b"
                bs.append(chunk)  # shape (..., rows)
        return Ws, bs

    def forward(self, flat, x):
        """
        flat: (B, param_dim)
        x:    (n, data_dim)
        returns: logits of shape (B, n, C)
        """
        B = flat.shape[0]
        out = []
        for b in range(B):
            W_list, b_list = self.unpack_params(flat[b:b+1])
            # unpack returns lists of length L+1
            W_list = [W_list[i][0] for i in range(len(W_list))]
            b_list = [b_list[i][0] for i in range(len(b_list))]

            h = x  # (n, data_dim)
            # hidden layers
            for i in range(self.L):
                z = h @ W_list[i].T + b_list[i]  # (n, h)
                h = np.maximum(z, 0)             # ReLU
            # final affine
            logits = h @ W_list[-1].T + b_list[-1]  # (n, C)
            out.append(logits)
        return np.stack(out, axis=0)  # (B, n, C)

    def compute_gradients(self, flat, x):
        """
        returns ∂ logits[b,i,k] / ∂ flat[b,:] of shape
          (B, n, C, param_dim)
        """
        def flat_to_logits(p):
            # returns (n*C,) flattened
            L = self.forward(p[None, :], x)[0]
            return L.reshape(-1)

        # build Jacobian for *one* parameter vector
        jac_single = jacobian(flat_to_logits)  # (param_dim, n*C)

        # manually batch it
        J_list = [jac_single(flat[b]) for b in range(flat.shape[0])]
        J = np.stack(J_list, axis=0)  # (B, param_dim, n*C)

        B = flat.shape[0]
        n = x.shape[0]
        return J.reshape(B, self.param_dim, n, self.C).transpose(0,2,3,1)

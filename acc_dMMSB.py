import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.special import logsumexp
from jax.nn import softmax

import tqdm

from jax import vmap, jit
from jax.tree_util import register_pytree_node_class
from functools import partial

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-8


@register_pytree_node_class
class dMMSB_State:
    """A PyTree class to hold the state of the dMMSB model."""
    def __init__(self, 
                 N, K, T, 
                 key=None, B=None, mu=None, Sigma=None, 
                 gamma_tilde=None, Sigma_tilde=None, 
                 delta=None, Phi=None, nu=None,
                 P=None, L=None, Y=None
                 ):
        # --- Static Configuration ---
        self.N = N
        self.K = K
        self.T = T

        # --- Dynamic Parameters (JAX arrays) ---
        self.key = key 
        self.B = B
        self.mu = mu
        self.Sigma = Sigma
        self.gamma_tilde = gamma_tilde
        self.Sigma_tilde = Sigma_tilde
        self.delta = delta
        self.Phi = Phi
        self.nu = nu
        self.P = P
        self.L = L
        self.Y = Y

    def tree_flatten(self):
        """Tells JAX how to flatten the object."""
        children = (self.key, self.B, self.mu, self.Sigma, 
                    self.gamma_tilde, self.Sigma_tilde, 
                    self.delta, self.Phi, self.nu,
                    self.P, self.L, self.Y)
        
        aux_data = {'N': self.N, 'K': self.K, 'T': self.T}
        
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Tells JAX how to unflatten the object."""
        key, B, mu, Sigma, gamma_tilde, Sigma_tilde, delta, Phi, nu, P, L, Y = children
        return cls(N=aux_data['N'], K=aux_data['K'], T=aux_data['T'], key=key, B=B, mu=mu, Sigma=Sigma, 
                   gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde, delta=delta, Phi=Phi, nu=nu,
                   P=P, L=L, Y=Y)

    def replace(self, **kwargs):
        """A helper method for immutably updating the state."""
        current_fields = {
            'key': self.key, 'B': self.B, 'mu': self.mu, 'Sigma': self.Sigma,
            'gamma_tilde': self.gamma_tilde, 'Sigma_tilde': self.Sigma_tilde, 'delta': self.delta,
            'Phi': self.Phi, 'nu': self.nu, 'P': self.P, 'L': self.L, 'Y': self.Y
        }
        updated_fields = {**current_fields, **kwargs}
        
        return dMMSB_State(N=self.N, K=self.K, T=self.T, **updated_fields)
    

def init_dMMSB_state(N, K, T, key=None, B=None, mu=None, Sigma=None, Phi=None, nu=None):
    """Initializes the dMMSB state with given or random parameters."""
    rng = key if key is not None else jax.random.PRNGKey(0)
    rng, b_key, mu_key, gamma_key, phi_key, nu_key = jax.random.split(rng, 6)
    if B is None:
        B = jax.random.uniform(b_key, (K, K)) #shape (K,K)
    if nu is None:
        nu = jax.random.normal(nu_key) # scalar
    if mu is None:
        mu = jnp.tile(jnp.zeros(K - 1)[None, :], (T, 1)) + nu  #shape (T, K-1)
    if Phi is None:
        Phi = jnp.eye(K - 1) * 20 #shape (K-1, K-1)
    if Sigma is None:
        Sigma = jnp.tile(jnp.eye(K - 1)[None, :, :], (T, 1, 1)) * 10 #shape (T, K-1, K-1)

    #KF RTS variables
    P = jnp.tile(jnp.eye(K - 1)[None, :, :], (T, 1, 1)) * 20 #shape (T, K-1, K-1) | NOTE: large initial covariance
    Y = None
    L = None

    return dMMSB_State(N=N, K=K, T=T, key=rng, B=B, mu=mu, Sigma=Sigma, Phi=Phi, nu=nu, P=P, Y=Y, L=L)


def _expand_gamma(gamma_km1, N):
    '''
    Help function to expand gamma from K-1 to K by appending zeros
    gamma_km1: (..., K-1) 
    returns: (..., K)
    '''
    zeros = jnp.zeros((N, 1))
    return jnp.concatenate([gamma_km1, zeros], axis=-1)

#--------------------------------------------------------------
# Inner Loop functions (From static version)
#--------------------------------------------------------------

def _compute_deltas(gamma_tilde, B, E):
    '''
    Compute delta matrices for all pairs (i,j) given current gamma_tilde and B
    gamma_tilde: (N,K) - NOTE: This should be K-dimensional (expanded)
    E: adjacency matrix (N,N)
    Returns: delta (N,N,K,K)
    '''
    
    gamma_i = gamma_tilde[:, None, :, None] # shape (N,1,K,1)
    gamma_j = gamma_tilde[None, :, None, :] # shape (1,N,1,K)

    gamma_sum = gamma_i + gamma_j # shape (N,N,K,K)

    B_reshaped = B[None, None, :, :] # shape (1,1,K,K)
    E_reshaped = E[:, :, None, None] # shape (N,N,1,1)

    log_bernoulli = jnp.where(E_reshaped == 1,
                            jnp.log(B_reshaped + EPS),
                            jnp.log1p(-B_reshaped + EPS))  # log(1-B) more stable

    delta_exp_term = gamma_sum + log_bernoulli # shape (N,N,K,K)
    max_delta_exp = jnp.max(delta_exp_term, axis=(-1,-2), keepdims=True)
    delta = jnp.exp(delta_exp_term - (max_delta_exp + logsumexp(delta_exp_term - max_delta_exp, axis=(-1,-2), keepdims=True))) # shape (N,N,K,K) logsumexp trick for numerical stability
    return delta # shape (N,N,K,K)

def _compute_g_H(gamma_hat, K):
    '''
    Compute g and H at gamma_hat
    gamma_hat: (N, K)
    Returns: g: (N, K), H: (N, K, K)
    '''
    
    max_gamma = jnp.max(gamma_hat, axis=-1, keepdims=True)
    g = jnp.exp(gamma_hat - (max_gamma + logsumexp(gamma_hat - max_gamma, axis=-1, keepdims=True))) # shape (N,K)
    H = jnp.einsum('ni,ij->nij', g, jnp.eye(K)) - jnp.einsum('ni,nj->nij', g, g) # shape (N,K,K)
    
    return g, H

def _update_sigma_tilde(Sigma_inv, H, N, K):
    '''
    Compute Sigma_tilde = (Sigma^{-1} + (2N-2) H_{1:K-1, 1:K-1})^{-1}
    Sigma_inv: (K-1, K-1)
    H: (N, K, K) Hessian from the K-dimensional expansion
    N: scalar, number of nodes
    K: scalar, number of roles
    Returns: Sigma_tilde: (N, K-1, K-1)
    '''
    # Take the top-left (K-1, K-1) block of the Hessian
    H_km1 = H[:, :K - 1, :K - 1] # shape (N, K-1, K-1)
    
    factor = 2.0 * N - 2.0
    A = Sigma_inv[None, :, :] + factor * H_km1
    jitter = EPS * jnp.eye(K - 1) # for numerical stability
    A = A + jitter[None, :, :]
    Sigma_tilde = jnp.linalg.inv(A)
    return Sigma_tilde

def _compute_m_expect(delta):
    '''
    Compute m_expect per node: m_i,k = sum_{j != i} (E[z_i->j,k] + E[z_i<-j,k])
    delta: (N,N,K,K)
    Returns: m_expect: (N,K)
    '''
    z_ij = jnp.sum(delta, axis=-1) # shape (N,N,K) Expected z_i->j (sender)
    z_ji = jnp.sum(delta, axis=-2) # shape (N,N,K) Expected z_i<-j (receiver)

    z_ij_expected = jnp.sum(z_ij, axis=1) # shape (N,K)
    z_ji_expected = jnp.sum(z_ji, axis=0) # shape (N,K)

    z_sum = z_ij_expected + z_ji_expected# shape (N,K)

    diag_ij = jnp.diagonal(z_ij, axis1=0, axis2=1).T # shape(N,K)
    diag_ji = jnp.diagonal(z_ji, axis1=0, axis2=1).T # shape(N,K)

    m_expect = z_sum - diag_ij - diag_ji # shape (N,K)

    return m_expect # shape (N,K)

def _update_gamma_tilde(delta, mu, Sigma_tilde, gamma_hat, g, H, N, K):
    '''
    Update gamma_tilde using Laplace approximation
    delta: (N,N,K,K)
    mu: (K-1,)
    Sigma_tilde: (N,K-1,K-1)
    gamma_hat: (N,K)
    g: (N,K)
    H: (N,K,K)
    N: scalar, number of nodes
    K: scalar, number of roles
    Returns: gamma_tilde: (N,K-1)
    '''
    factor = 2.0 * N - 2.0

    # m_expect is computed over all K roles and must not be truncated.
    m_expect = _compute_m_expect(delta) # shape (N, K)
    
    # Expand mu to K dimensions for the calculation
    mu_expanded = jnp.append(mu, 0.0) # shape (K,)
    
    # The term in brackets from the paper's appendix is K-dimensional
    # It uses all K-dimensional components: m_expect, g, H, gamma_hat, mu_expanded
    term_1_full = (m_expect - factor * g + 
                factor * jnp.einsum('nij,nj->ni', H, gamma_hat) - 
                factor * jnp.einsum('nij,j->ni', H, mu_expanded)) # shape (N, K)

    # Truncate the update vector to K-1 dimensions right before the final multiplication
    term_1_km1 = term_1_full[:, :K - 1] # shape (N, K-1)

    # Final update is in K-1 dimensional space
    gamma_tilde = mu[None, :] + jnp.einsum('nij,nj->ni', Sigma_tilde, term_1_km1) # shape (N, K-1)

    return gamma_tilde # shape (N, K-1)

def _inner_step_static(gamma_tilde, Sigma_tilde, mu, Sigma_inv, B, E, N, K):
    '''
    Perform one inner iteration to update gamma_tilde and Sigma_tilde
    gamma_tilde: (N,K-1)
    Sigma_tilde: (N,K-1,K-1)
    mu: (K-1,)
    Sigma_inv: (K-1,K-1)
    B: (K,K)
    E: adjacency matrix (N,N)
    Returns: updated gamma_tilde (N,K-1), Sigma_tilde (N,K-1,K-1), delta (N,N,K,K)
    '''
    # Expand gamma_tilde to K dimensions for delta computation
    gamma_hat = _expand_gamma(gamma_tilde, N) # shape (N, K)

    delta = _compute_deltas(gamma_hat, B, E) # shape (N,N,K,K)
    g, H = _compute_g_H(gamma_hat, K) # g: (N,K), H: (N,K,K)
    Sigma_tilde = _update_sigma_tilde(Sigma_inv, H, N, K) # shape (N,K-1,K-1)
    gamma_tilde = _update_gamma_tilde(delta, mu, Sigma_tilde, gamma_hat, g, H, N, K) # shape (N,K-1)
    return gamma_tilde, Sigma_tilde, delta

#--------------------------------------------------------------

def _inner_step(gamma_tilde, Sigma_tilde, mu, Sigma_inv, B, E, N, K):
    '''
    Perform one inner iteration to update gamma_tilde and Sigma_tilde for all time steps using vmap
    gamma_tilde: (T,N,K-1)
    Sigma_tilde: (T,N,K-1,K-1)
    mu: (T,K-1)
    Sigma_inv: (T,K-1,K-1)
    B: (K,K)
    E: adjacency matrix (T,N,N)
    N: scalar, number of nodes
    Returns: updated 
    gamma_tilde: (T,N,K-1)
    Sigma_tild: (T,N,K-1,K-1)
    delta: (T,N,N,K,K)
    '''

    gamma_tilde, Sigma_tilde, delta = vmap(_inner_step_static, in_axes=(0,0,0,0,None,0,None,None))(gamma_tilde, Sigma_tilde, mu, Sigma_inv, B, E, N, K)
    return gamma_tilde, Sigma_tilde, delta

def _update_mu_P_L(mu, P, Y, Sigma, Phi, N):
    '''
    Update mu using Kalman filter and RTS smoother with jax.lax.scan. eq (14,15,16,17)
    mu: shape (T,K-1) 
    P: shape (T,K-1,K-1)
    Y: shape (T,K-1)
    Sigma: shape (T,K-1,K-1) 
    Phi: shape (K-1,K-1) 
    N: scalar, number of nodes
    '''

    # --- 1. Kalman Filter (Forward Pass) ---
    def kalman_step(carry, inputs):
        mu_prev, P_prev = carry
        Y_t, Sigma_t = inputs

        # Prediction step
        mu_pred_t = mu_prev
        P_pred_t = P_prev + Phi

        # Update step
        tmp = P_pred_t + Sigma_t/N
        K_t = jnp.linalg.solve(tmp.T, P_pred_t.T).T  #numerically stable version
        mu_t = mu_pred_t + K_t @ (Y_t - mu_pred_t)
        P_t = P_pred_t - K_t @ P_pred_t

        new_carry = (mu_t, P_t)
        # Stack filtered and predicted states for the backward pass
        outputs_to_stack = (mu_t, P_t, mu_pred_t, P_pred_t)
        return new_carry, outputs_to_stack

    init_carry = (mu[0], P[0])
    inputs = (Y[1:], Sigma[1:])
    _, (mu_filtered_scanned, P_filtered_scanned, mu_pred_scanned, P_pred_scanned) = jax.lax.scan(
        kalman_step, init_carry, inputs, unroll=True
    )

    # Combine initial state with scanned results
    mu_filtered = jnp.concatenate([mu[0][None, :], mu_filtered_scanned], axis=0)
    P_filtered = jnp.concatenate([P[0][None, :, :], P_filtered_scanned], axis=0)
    # The prediction for time t is mu_{t-1}, so mu_pred starts from mu_0
    mu_pred = jnp.concatenate([mu[0][None, :], mu_pred_scanned], axis=0)
    P_pred = jnp.concatenate([P[0][None, :, :], P_pred_scanned], axis=0)

    # --- 2. RTS Smoother (Backward Pass) ---
    def rts_smoother_step(carry, inputs):
        mu_smooth_next, P_smooth_next = carry
        mu_filtered_t, P_filtered_t, mu_pred_next, P_pred_next = inputs

        L_t = jnp.linalg.solve(P_pred_next.T, P_filtered_t.T).T  # numerically stable version
        
        # Update step
        mu_smooth_t = mu_filtered_t + L_t @ (mu_smooth_next - mu_pred_next)
        P_smooth_t = P_filtered_t + L_t @ (P_smooth_next - P_pred_next) @ L_t.T

        new_carry = (mu_smooth_t, P_smooth_t)
        outputs_to_stack = (mu_smooth_t, P_smooth_t, L_t)
        return new_carry, outputs_to_stack

    init_carry_smooth = (mu_filtered[-1], P_filtered[-1])
    inputs_smooth = (mu_filtered[:-1], P_filtered[:-1], mu_pred[1:], P_pred[1:])
    
    _, (mu_smooth_scanned, P_smooth_scanned, L_scanned) = jax.lax.scan(
        rts_smoother_step, init_carry_smooth, inputs_smooth, reverse=True, unroll=True
    )

    mu_smooth = jnp.concatenate([mu_smooth_scanned, mu_filtered[-1][None, :]], axis=0)
    P_smooth = jnp.concatenate([P_smooth_scanned, P_filtered[-1][None, :, :]], axis=0)
    L = jnp.concatenate([L_scanned, jnp.zeros((1, P.shape[1], P.shape[2]))], axis=0)  # Last L is not defined

    return mu_smooth, P_smooth, L 

def _update_B(delta, E):
    '''
    Update B using the current parameters. eq (26)
    delta: shape (T,N,N,K,K)
    E: shape (T,N,N)
    '''
    E_reshaped = E[:, :, :, None, None] # shape (T,N,N,1,1)
    
    num = jnp.sum(delta * E_reshaped, axis=(0,1,2)) # shape (K,K)
    den = jnp.sum(delta, axis=(0,1,2)) # shape (K,K)

    B_new = jnp.where(den < EPS, 
                0.5 * jnp.ones_like(num),  
                num / jnp.maximum(den, EPS)) # shape (K,K)
    B_new = jnp.clip(B_new, 1e-6, 1 - 1e-6)

    return B_new

def _update_Phi(mu, P, L, K):
    """
    Vectorized update of Phi using eq (19).
    mu: shape (T, K-1)
    P: shape (T, K-1, K-1)
    L: shape (T, K-1, K-1)
    """
    diffs = mu[1:] - mu[:-1]  

    term1 = jnp.einsum("ti,tj->tij", diffs, diffs) # shape (T-1, K-1, K-1)
    term2 = jnp.einsum("tik,tkl,tjl->tij", L[:-1], P[1:], L[:-1]) # shape (T-1, K-1, K-1)

    Phi_new = (term1 + term2).mean(axis=0)
    # Ensure positive definiteness
    Phi_new = Phi_new + jnp.eye(K - 1) * EPS  
    return Phi_new

def _update_Sigma(mu, gamma_tilde, Sigma_tilde, N, K):
    '''
    Update Sigma_tilde using the current parameters. eq (20)
    mu: shape (T,K-1)
    gamma_tilde: shape (T,N,K-1)
    Sigma_tilde: shape (T,N,K-1,K-1)
    N: scalar, number of nodes
    '''
    
    diff = mu[:, None, :] - gamma_tilde # shape (T,N,K-1)
    
    sum_outer_products = jnp.einsum('tnk,tnj->tkj', diff, diff)  # shape (T,K-1,K-1) 

    sum_Sigma_tilde = jnp.sum(Sigma_tilde, axis=1)  # shape (T,K-1,K-1)

    Sigma_new = (sum_outer_products + sum_Sigma_tilde) / N  # shape (T,K-1,K-1)

    Sigma_new = Sigma_new + jnp.eye(K - 1)[None, :, :] * EPS  # Ensure positive definiteness
    return Sigma_new

def _update_nu(mu):
    '''
    Update nu using the current parameters. eq (21)
    mu: shape (T,K-1)
    '''
    return mu[0]  # Return first time step mu (still K-1 dimensional)

def init_q_gamma(state: dMMSB_State):
    '''
    Initialize q(gamma) and Sigma^-1 for all time steps.
    state: dMMSB_State
    Returns: gamma_tilde (T,N,K-1), Sigma_tilde (T,N,K-1,K-1), Sigma_inv (T,K-1,K-1)
    '''
    def init_q_gamma_static(key, mu_t, Sigma_t, N, K):
                '''
                Initialize q(gamma) and Sigma^-1 for a single time step.
                mu_t: (K-1,)
                Sigma_t: (K-1,K-1)
                Returns: new state, Sigma_inv (K-1,K-1)
                '''
                key, subkey = jax.random.split(key)
                gamma_tilde = jax.random.multivariate_normal(subkey, mu_t, Sigma_t, shape=(N,)) # shape (N,K-1)
                gamma_tilde = jnp.clip(gamma_tilde, -10, 10)  # gradient clipping
                # Expand for g,H computation
                gamma_hat = _expand_gamma(gamma_tilde, N) # shape (N, K)
                g, H = _compute_g_H(gamma_hat, K) # g: (N,K), H: (N,K,K)

                jitter = 1e-5 * jnp.eye(K - 1) # for numerical stability
                Sigma_inv = jnp.linalg.inv(Sigma_t + jitter) # shape (K-1,K-1)
                Sigma_tilde = _update_sigma_tilde(Sigma_inv, H, N, K) # shape (N,K-1,K-1)

                return gamma_tilde, Sigma_tilde, Sigma_inv
    
    # Initialize delta with zeros of the correct shape before the loop
    delta_init = jnp.zeros((state.T, state.N, state.N, state.K, state.K))

    gamma_tilde, Sigma_tilde, Sigma_inv = vmap(init_q_gamma_static, in_axes=(None, 0, 0, None, None))(state.key, state.mu, state.Sigma, state.N, state.K) # shape (T,N,K-1), (T,N,K-1,K-1), (T,K-1,K-1)

    return state.replace(gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde, delta=delta_init), Sigma_inv # Update state with new key and initialized values



# --- Jitted Step Functions ---

def e_step_inner_updates(state: dMMSB_State, Sigma_inv, E):
    """Jitted E-step inner loop updates."""
    gamma_tilde_new, Sigma_tilde_new, delta_new = _inner_step(state.gamma_tilde, state.Sigma_tilde, state.mu, Sigma_inv, state.B, E, state.N, state.K)
   
    B_new = _update_B(delta_new, E) 

    return state.replace(gamma_tilde=gamma_tilde_new, Sigma_tilde=Sigma_tilde_new, delta=delta_new, B=B_new)

@jit   
def inner_loop(state: dMMSB_State, E, max_inner_iters=100, tol=1e-6):
    """Run the inner loop of the E-step until convergence."""
    i = 0 
    d_ll = jnp.inf
    prev_ll = -jnp.inf

    state, Sigma_inv = init_q_gamma(state) # Re-initialize q(gamma) at the start of the inner loop
 
    def cond_fn(carry):
        state, i, d_ll, prev_ll = carry
        return (d_ll > tol) & (i < max_inner_iters)

    def body_fn(carry):
        state, i, d_ll, prev_ll = carry
        state = e_step_inner_updates(state, Sigma_inv, E)

        i += 1
        ll = log_likelihood(state.delta, state.B, E)
        d_ll = jnp.abs(ll - prev_ll)
        prev_ll = ll

        #jax.debug.print("  [inner {}/{}] ll: {:.4f}, d_ll: {:.6f}", i, max_inner_iters, ll, d_ll)
        return state, i, d_ll, prev_ll
    
    init_carry = (state, i, d_ll, prev_ll)

    state, i, d_ll, prev_ll = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return state, prev_ll

@jit
def m_step_outer_updates(state: dMMSB_State):
    """Jitted M-step outer loop updates."""
    Y = jnp.mean(state.gamma_tilde, axis=1)
    mu, P, L = _update_mu_P_L(state.mu, state.P, Y, state.Sigma, state.Phi, state.N)
    nu = _update_nu(mu)
    Phi = _update_Phi(mu, P, L, state.K)
    Sigma = _update_Sigma(mu, state.gamma_tilde, state.Sigma_tilde, state.N, state.K)
    return state.replace(Y=Y, mu=mu, P=P, L=L, nu=nu, Phi=Phi, Sigma=Sigma)

@jit    
def log_likelihood(delta, B, E):
    '''
    Compute the log likelihood of the data given the current parameters. eq (25)
    delta: shape (T,N,N,K,K)
    B: shape (K,K)
    E: shape (T,N,N)
    '''
    E_reshaped = E[:, :, :, None, None] # shape (T,N,N,1,1)
    B_reshaped = B[None, None, None, :, :] # shape (1,1,1,K,K)
    logB = jnp.log(B_reshaped + EPS) # shape (1,1,1,K,K)
    log1mB = jnp.log(1.0 - B_reshaped + EPS) # shape (1,1,1,K,K)

    ll_matrix = delta * (E_reshaped * logB + (1.0 - E_reshaped) * log1mB) # shape (T,N,N,K,K)
    ll = jnp.sum(ll_matrix)
    return ll

# --- Public API ---

class jitdMMSB:
    """A JAX-accelerated implementation of the dMMSB model."""

    def __init__(self, nodes, roles, timesteps, **kwargs):
        self.N = nodes
        self.K = roles
        self.T = timesteps
        key_seed = kwargs.pop('key', 0)
        self.state = init_dMMSB_state(self.N, self.K, self.T, key=key_seed, **kwargs)

    def fit(self, E, max_inner_iters=100, max_outer_iters=100, tol=1e-6, verbose=False):
        i = 0 
        d_ll = jnp.inf
        prev_outer_ll = -jnp.inf


        while(d_ll > tol and i < max_outer_iters):
            
            # Initialize q(gamma) for the E-step   
            self.state, Sigma_inv = init_q_gamma(self.state)

            # Inner Loop

            self.state, inner_ll = inner_loop(self.state, E, max_inner_iters, tol)
            # j = 0
            # inner_d_ll = jnp.inf
            # prev_inner_ll = -jnp.inf
            # while(inner_d_ll > tol and j < max_inner_iters):
            #     self.state = e_step_inner_updates(self.state, Sigma_inv, E)

            #     j += 1
            #     inner_ll = log_likelihood(self.state.delta, self.state.B, E)
            #     inner_d_ll = jnp.abs(inner_ll - prev_inner_ll)
            #     prev_inner_ll = inner_ll
            #     if verbose:
            #         print(f"  [inner {j}] ll: {inner_ll:.4f}, d_ll: {inner_d_ll:.6f}")
            
            # M-Step
            self.state = m_step_outer_updates(self.state)

            # Convergence check
            i += 1
            outer_ll = inner_ll
            d_ll = jnp.abs(outer_ll - prev_outer_ll)
            prev_outer_ll = outer_ll
            if verbose:
                print(f"[outer {i}] ll: {outer_ll:.4f}, d_ll: {d_ll:.6f}")
        return outer_ll
    
    def generate_graph(self):
        '''
        Generate a graph from the model
        Returns: adjacency matrix E (T,N,N)
        '''
        if self.state.B is None:
            raise ValueError("Must initialize B before generating a graph.")

        key = self.state.key
        if self.state.gamma_tilde is None:
            if self.state.nu is None or self.state.Phi is None or self.state.Sigma is None:
                raise ValueError("Must initialize nu, Phi, and Sigma before generating a graph if gamma_tilde is not set.")
            
            print("mu is none")
            mu = jnp.zeros((self.state.T, self.state.K - 1))
            mean = self.state.nu 
            key, subkey = jax.random.split(key)
            mu_0 = jax.random.multivariate_normal(subkey, mean, self.state.Phi)
            mu = mu.at[0].set(mu_0)
        
            for t in range(1, self.T):
                key, subkey = jax.random.split(key)
                mu_t = jax.random.multivariate_normal(subkey, mu[t-1], self.Phi)
                mu = self.mu.at[t].set(mu_t)
            
            gamma_tilde = jnp.zeros((self.T, self.N, self.K - 1))
            for t in range(self.T):
                key, subkey = jax.random.split(key)
                gamma_t = jax.random.multivariate_normal(subkey,mu[t], self.state.Sigma[t], shape=(self.state.N,))
                gamma_tilde = self.gamma_tilde.at[t].set(gamma_t)
        
            self.state = self.state.replace(gamma_tilde=gamma_tilde, mu=mu, key=key)
        gamma_expanded = vmap(_expand_gamma, in_axes=(0, None))(self.state.gamma_tilde, self.state.N) # shape (T,N,K)
        pis = softmax(gamma_expanded, axis=-1) # shape (T,N,K)

        def sample_E(key, pis_t):
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            z_ij = jax.random.multinomial(subkey1, 1, pis_t, shape=(self.state.N, self.state.N, self.state.K)) # shape (N,N,K)
            z_ji = jax.random.multinomial(subkey2, 1, pis_t, shape=(self.state.N, self.state.N, self.state.K)) # shape (N,N,K)

            p = jnp.einsum('ijk, kl -> ijl', z_ij, self.B) # shape (N,N,K)
            p = jnp.einsum('ijl, jil -> ij', p, z_ji) # shape (N,N)
            E_t = jax.random.bernoulli(subkey3, p=p)
            return E_t
    
        E = vmap(sample_E, in_axes=(0,0))(jax.random.split(key, self.state.T), pis) # shape (T,N,N)
        return E.astype(jnp.int32) # shape (T,N,N)

    # Accessor properties
    @property
    def B(self): return self.state.B
    @property
    def mu(self): return self.state.mu
    @property
    def Sigma(self): return self.state.Sigma
    @property
    def gamma_tilde(self): return self.state.gamma_tilde
    @property
    def Sigma_tilde(self): return self.state.Sigma_tilde
    @property
    def nu(self): return self.state.nu
    @property
    def Phi(self): return self.state.Phi
    @property
    def key(self): return self.state.key

# import jax
# import jax.numpy as jnp
# import jax.scipy as jsp
# from jax.scipy.special import logsumexp
# from jax.nn import softmax

# from jax import vmap, jit
# from jax.tree_util import register_pytree_node_class
# from functools import partial

# EPS = 1e-12

# @register_pytree_node_class
# class dMMSB_State:
#     """A PyTree class to hold the state of the dMMSB model."""
#     def __init__(self, 
#                  N, K, T, 
#                  key=None, B=None, mu=None, Sigma=None, 
#                  gamma_tilde=None, Sigma_tilde=None, 
#                  delta=None, Phi=None, nu=None,
#                  P=None, L=None, Y=None
#                  ):
#         # --- Static Configuration ---
#         self.N = N
#         self.K = K
#         self.T = T

#         # --- Dynamic Parameters (JAX arrays) ---
#         self.key = key 
#         self.B = B
#         self.mu = mu
#         self.Sigma = Sigma
#         self.gamma_tilde = gamma_tilde
#         self.Sigma_tilde = Sigma_tilde
#         self.delta = delta
#         self.Phi = Phi
#         self.nu = nu
#         self.P = P
#         self.L = L
#         self.Y = Y

#     def tree_flatten(self):
#         """Tells JAX how to flatten the object."""
#         children = (self.key, self.B, self.mu, self.Sigma, 
#                     self.gamma_tilde, self.Sigma_tilde, 
#                     self.delta, self.Phi, self.nu,
#                     self.P, self.L, self.Y)
        
#         aux_data = {'N': self.N, 'K': self.K, 'T': self.T}
        
#         return (children, aux_data)

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         """Tells JAX how to unflatten the object."""
#         key, B, mu, Sigma, gamma_tilde, Sigma_tilde, delta, Phi, nu, P, L, Y = children
#         return cls(N=aux_data['N'], K=aux_data['K'], T=aux_data['T'], key=key, B=B, mu=mu, Sigma=Sigma, 
#                    gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde, delta=delta, Phi=Phi, nu=nu,
#                    P=P, L=L, Y=Y)

#     def replace(self, **kwargs):
#         """A helper method for immutably updating the state."""
#         current_fields = {
#             'key': self.key, 'B': self.B, 'mu': self.mu, 'Sigma': self.Sigma,
#             'gamma_tilde': self.gamma_tilde, 'Sigma_tilde': self.Sigma_tilde, 'delta': self.delta,
#             'Phi': self.Phi, 'nu': self.nu, 'P': self.P, 'L': self.L, 'Y': self.Y
#         }
#         updated_fields = {**current_fields, **kwargs}
        
#         return dMMSB_State(N=self.N, K=self.K, T=self.T, **updated_fields)
    

# def init_dMMSB_state(N, K, T, key=0, B=None, mu=None, Sigma=None, Phi=None, nu=None):
#     """Initializes the dMMSB state with given or random parameters."""
#     rng, b_key, mu_key, phi_key, nu_key = jax.random.split(key, 5)

#     if B is None:
#         B = jax.random.uniform(b_key, (K, K))
#     if nu is None:
#         nu = jax.random.normal(nu_key, (K - 1,))
#     if mu is None:
#         mu = jnp.tile(nu[None, :], (T, 1))
#     if Phi is None:
#         Phi = jnp.eye(K - 1) * 20
#     if Sigma is None:
#         Sigma = jnp.tile(jnp.eye(K - 1)[None, :, :], (T, 1, 1)) * 10

#     P = jnp.tile(jnp.eye(K - 1)[None, :, :], (T, 1, 1)) * 20
    
#     return dMMSB_State(N=N, K=K, T=T, key=rng, B=B, mu=mu, Sigma=Sigma, Phi=Phi, nu=nu, P=P, 
#                        gamma_tilde=None, Sigma_tilde=None, delta=None, Y=None, L=None)


# def _expand_gamma(gamma_km1, N):
#     """Expands gamma from K-1 to K dimensions by appending zeros."""
#     zeros = jnp.zeros((N, 1))
#     return jnp.concatenate([gamma_km1, zeros], axis=-1)
    
# @jit
# def log_likelihood(delta, B, E):
#     E_reshaped = E[:, :, :, None, None]
#     B_reshaped = B[None, None, None, :, :]
#     logB = jnp.log(B_reshaped + EPS)
#     log1mB = jnp.log(1.0 - B_reshaped + EPS)
#     ll_matrix = delta * (E_reshaped * logB + (1.0 - E_reshaped) * log1mB)
#     return jnp.sum(ll_matrix)

# #--------------------------------------------------------------
# # Inner Loop functions (Pure)
# #--------------------------------------------------------------

# def _compute_deltas(gamma_tilde_k, B, E, K):
#     gamma_i = gamma_tilde_k[:, None, :, None]
#     gamma_j = gamma_tilde_k[None, :, None, :]
#     gamma_sum = gamma_i + gamma_j
#     B_reshaped = B[None, None, :, :]
#     E_reshaped = E[:, :, None, None]
#     log_bernoulli = jnp.where(E_reshaped == 1, jnp.log(B_reshaped + EPS), jnp.log1p(-B_reshaped + EPS))
#     delta_exp_term = gamma_sum + log_bernoulli
#     max_delta_exp = jnp.max(delta_exp_term, axis=(-1,-2), keepdims=True)
#     delta = jnp.exp(delta_exp_term - (max_delta_exp + logsumexp(delta_exp_term - max_delta_exp, axis=(-1,-2), keepdims=True)))
#     return delta

# def _compute_g_H(gamma_hat, K):
#     max_gamma = jnp.max(gamma_hat, axis=-1, keepdims=True)
#     g = jnp.exp(gamma_hat - (max_gamma + logsumexp(gamma_hat - max_gamma, axis=-1, keepdims=True)))
#     H = jnp.einsum('ni,ij->nij', g, jnp.eye(K)) - jnp.einsum('ni,nj->nij', g, g)
#     return g, H

# def _update_sigma_tilde(Sigma_inv, H, N, K):
#     H_km1 = H[:, :K - 1, :K - 1]
#     factor = 2.0 * N - 2.0
#     A = Sigma_inv[None, :, :] + factor * H_km1
#     jitter = 1e-5 * jnp.eye(K - 1)
#     return jnp.linalg.inv(A + jitter[None, :, :])

# def _compute_m_expect(delta):
#     z_ij = jnp.sum(delta, axis=-1)
#     z_ji = jnp.sum(delta, axis=-2)
#     z_ij_expected = jnp.sum(z_ij, axis=1)
#     z_ji_expected = jnp.sum(z_ji, axis=0)
#     z_sum = z_ij_expected + z_ji_expected
#     diag_ij = jnp.diagonal(z_ij, axis1=0, axis2=1).T
#     diag_ji = jnp.diagonal(z_ji, axis1=0, axis2=1).T
#     return z_sum - diag_ij - diag_ji

# def _update_gamma_tilde(delta, mu, Sigma_tilde, gamma_hat, g, H, N, K):
#     '''
#     Update gamma_tilde using Laplace approximation
#     delta: (N,N,K,K)
#     mu: (K-1,)
#     Sigma_tilde: (N,K-1,K-1)
#     gamma_hat: (N,K)
#     g: (N,K)
#     H: (N,K,K)
#     Returns: gamma_tilde: (N,K-1)
#     '''
#     factor = 2.0 * N - 2.0

#     # m_expect is computed over all K roles and must not be truncated.
#     m_expect = _compute_m_expect(delta) # shape (N, K)
    
#     # Expand mu to K dimensions for the calculation
#     mu_expanded = jnp.append(mu, 0.0) # shape (K,)
    
#     # The term in brackets from the paper's appendix is K-dimensional
#     # It uses all K-dimensional components: m_expect, g, H, gamma_hat, mu_expanded
#     term_1_full = (m_expect - factor * g + 
#                 factor * jnp.einsum('nij,nj->ni', H, gamma_hat) - 
#                 factor * jnp.einsum('nij,j->ni', H, mu_expanded)) # shape (N, K)

#     # Truncate the update vector to K-1 dimensions right before the final multiplication
#     term_1_km1 = term_1_full[:, :K - 1] # shape (N, K-1)

#     # Final update is in K-1 dimensional space
#     gamma_tilde = mu[None, :] + jnp.einsum('nij,nj->ni', Sigma_tilde, term_1_km1) # shape (N, K-1)

#     return gamma_tilde # shape (N, K-1)

# def _inner_step_static(gamma_tilde_km1, mu_km1, Sigma_inv_km1, B, E, N, K):
#     gamma_hat_k = _expand_gamma(gamma_tilde_km1, N)
#     delta = _compute_deltas(gamma_hat_k, B, E, K)
#     g, H = _compute_g_H(gamma_hat_k, K)
#     Sigma_tilde_km1 = _update_sigma_tilde(Sigma_inv_km1, H, N, K)
#     gamma_tilde_km1_new = _update_gamma_tilde(delta, mu_km1, Sigma_tilde_km1, gamma_hat_k, g, H, N, K)
#     return gamma_tilde_km1_new, Sigma_tilde_km1, delta

# #--------------------------------------------------------------
# # M-Step and Outer Loop Functions (Pure)
# #--------------------------------------------------------------

# def _update_mu_P_L(mu, P, Y, Sigma, Phi, N, T, K):
#     def kalman_step(carry, inputs):
#         mu_prev, P_prev = carry
#         Y_t, Sigma_t = inputs
#         mu_pred_t = mu_prev
#         P_pred_t = P_prev + Phi
#         tmp = P_pred_t + Sigma_t / N
#         K_t = jnp.linalg.solve(tmp.T, P_pred_t.T).T
#         mu_t = mu_pred_t + K_t @ (Y_t - mu_pred_t)
#         P_t = P_pred_t - K_t @ P_pred_t
#         return (mu_t, P_t), (mu_t, P_t, mu_pred_t, P_pred_t)

#     init_carry = (mu[0], P[0])
#     _, (mu_f_scan, P_f_scan, mu_p_scan, P_p_scan) = jax.lax.scan(kalman_step, init_carry, (Y[1:], Sigma[1:]))
#     mu_filtered = jnp.concatenate([mu[0][None, :], mu_f_scan])
#     P_filtered = jnp.concatenate([P[0][None, :, :], P_f_scan])
#     mu_pred = jnp.concatenate([mu[0][None, :], mu_p_scan])
#     P_pred = jnp.concatenate([P[0][None, :, :], P_p_scan])

#     def rts_step(carry, inputs):
#         mu_s_next, P_s_next = carry
#         mu_f_t, P_f_t, mu_p_next, P_p_next = inputs
#         L_t = jnp.linalg.solve(P_p_next.T, P_f_t.T).T
#         mu_s_t = mu_f_t + L_t @ (mu_s_next - mu_p_next)
#         P_s_t = P_f_t + L_t @ (P_s_next - P_p_next) @ L_t.T
#         return (mu_s_t, P_s_t), (mu_s_t, P_s_t, L_t)

#     init_carry_smooth = (mu_filtered[-1], P_filtered[-1])
#     _, (mu_s_scan, P_s_scan, L_scan) = jax.lax.scan(rts_step, init_carry_smooth, 
#                                                    (mu_filtered[:-1], P_filtered[:-1], mu_pred[1:], P_pred[1:]), 
#                                                    reverse=True)
#     mu_smooth = jnp.concatenate([mu_s_scan, mu_filtered[-1][None, :]])
#     P_smooth = jnp.concatenate([P_s_scan, P_filtered[-1][None, :, :]])
#     L = jnp.concatenate([L_scan, jnp.zeros((1, K - 1, K - 1))])
#     return mu_smooth, P_smooth, L

# def _update_B(delta, E):
#     E_reshaped = E[:, :, :, None, None]
#     num = jnp.sum(delta * E_reshaped, axis=(0, 1, 2))
#     den = jnp.sum(delta, axis=(0, 1, 2))
#     B_new = jnp.where(den < EPS, 0.5 * jnp.ones_like(num), num / jnp.maximum(den, EPS))
#     return jnp.clip(B_new, EPS, 1 - EPS)

# def _update_Phi(mu, P, L, K):
#     diffs = mu[1:] - mu[:-1]
#     term1 = jnp.einsum("ti,tj->tij", diffs, diffs)
#     term2 = jnp.einsum("tik,tkl,tjl->tij", L[:-1], P[1:], L[:-1])
#     Phi_new = (term1 + term2).mean(axis=0)
#     return Phi_new + jnp.eye(K - 1) * EPS

# def _update_Sigma(mu, gamma_tilde, Sigma_tilde, N, K):
#     diff = mu[:, None, :] - gamma_tilde
#     sum_outer_products = jnp.einsum('tnk,tnj->tkj', diff, diff)
#     sum_Sigma_tilde = jnp.sum(Sigma_tilde, axis=1)
#     Sigma_new = (sum_outer_products + sum_Sigma_tilde) / N
#     return Sigma_new + jnp.eye(K - 1)[None, :, :] * 1e-5

# def _update_nu(mu):
#     return mu[0]

# # --- Jitted Step Functions ---
# @partial(jit, static_argnames=['N', 'K'])
# def e_step_inner_updates(state, Sigma_inv, E, N, K):
#     """Jitted E-step inner loop updates."""
#     gamma_tilde_new, Sigma_tilde_new, delta_new = vmap(
#         _inner_step_static, in_axes=(0, 0, 0, None, 0, None, None)
#     )(state.gamma_tilde, state.mu, Sigma_inv, state.B, E, N, K)
   
#     B_new = _update_B(delta_new, E) 

#     return state.replace(gamma_tilde=gamma_tilde_new, Sigma_tilde=Sigma_tilde_new, delta=delta_new, B=B_new)
    


# @partial(jit, static_argnames=['N', 'T', 'K'])
# def m_step_outer_updates(state, N, T, K):
#     """Jitted M-step outer loop updates."""
#     Y = jnp.mean(state.gamma_tilde, axis=1)
#     mu, P, L = _update_mu_P_L(state.mu, state.P, Y, state.Sigma, state.Phi, N, T, K)
#     nu = _update_nu(mu)
#     Phi = _update_Phi(mu, P, L, K)
#     Sigma = _update_Sigma(mu, state.gamma_tilde, state.Sigma_tilde, N, K)
#     return state.replace(Y=Y, mu=mu, P=P, L=L, nu=nu, Phi=Phi, Sigma=Sigma)

# # --- Public API ---

# class jitdMMSB:
#     """A JAX-accelerated implementation of the dMMSB model."""

#     def __init__(self, nodes, roles, timesteps, **kwargs):
#         self.N = nodes
#         self.K = roles
#         self.T = timesteps
#         key_seed = kwargs.pop('key', 0)
#         self.state = init_dMMSB_state(self.N, self.K, self.T, key=key_seed, **kwargs)

#     def fit(self, E, max_inner_iters=100, max_outer_iters=100, tol=1e-6, verbose=False):
#         i = 0 
#         d_ll = jnp.inf
#         prev_outer_ll = -jnp.inf


#         while(d_ll > tol and i < max_outer_iters):
#             # Initialize q(gamma) for the E-step
#             key, subkey = jax.random.split(self.state.key)

#             def init_q_gamma(key, mu_t, Sigma_t, N, K):
#                 '''
#                 Initialize q(gamma) and Sigma^-1 for a single time step.
#                 mu_t: (K-1,)
#                 Sigma_t: (K-1,K-1)
#                 Returns: gamma_tilde (N,K-1), Sigma_tilde (N,K-1,K-1), Sigma_inv (K-1,K-1)
#                 '''
#                 key, subkey = jax.random.split(key)
#                 gamma_tilde = jax.random.multivariate_normal(subkey, mu_t, Sigma_t, shape=(N,)) # shape (N,K-1)
#                 gamma_tilde = jnp.clip(gamma_tilde, -10, 10)  # gradient clipping
#                 # Expand for g,H computation
#                 gamma_hat = _expand_gamma(gamma_tilde, N) # shape (N, K)
#                 g, H = _compute_g_H(gamma_hat, K) # g: (N,K), H: (N,K,K)

#                 jitter = 1e-5 * jnp.eye(K - 1) # for numerical stability
#                 Sigma_inv = jnp.linalg.inv(Sigma_t + jitter) # shape (K-1,K-1)
#                 Sigma_tilde = _update_sigma_tilde(Sigma_inv, H, N, K) # shape (N,K-1,K-1)

#                 return gamma_tilde, Sigma_tilde, Sigma_inv

#             gamma_tilde, Sigma_tilde, Sigma_inv = vmap(init_q_gamma, in_axes=(None, 0, 0, None, None))(key, self.state.mu, self.state.Sigma, self.state.N, self.state.K) # shape (T,N,K-1), (T,N,K-1,K-1), (T,K-1,K-1)

#             self.state = self.state.replace(key=key, gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde) # Update state with new key and initialized values

#             # Inner Loop
#             j = 0
#             inner_d_ll = jnp.inf
#             prev_inner_ll = -jnp.inf
#             while(inner_d_ll > tol and j < max_inner_iters):
#                 self.state = e_step_inner_updates(self.state, Sigma_inv, E, self.N, self.K)

#                 j += 1
#                 inner_ll = log_likelihood(self.state.delta, self.state.B, E)
#                 inner_d_ll = jnp.abs(inner_ll - prev_inner_ll)
#                 prev_inner_ll = inner_ll
#                 if verbose:
#                     print(f"  [inner {j}] ll: {inner_ll:.4f}, d_ll: {inner_d_ll:.6f}")
            
#             # M-Step
#             self.state = m_step_outer_updates(self.state, self.N, self.T, self.K)

#             # Convergence check
#             i += 1
#             outer_ll = inner_ll
#             d_ll = jnp.abs(outer_ll - prev_outer_ll)
#             prev_outer_ll = outer_ll
#             if verbose:
#                 print(f"[outer {i}] ll: {outer_ll:.4f}, d_ll: {d_ll:.6f}")
#         return outer_ll
    
#     def generate_graph(self):
#         """Generate a graph from the model."""
#         key, mu_key, gamma_key, e_key = jax.random.split(self.state.key, 4)
        
#         # 1. Generate mu sequence
#         mu_t = jnp.zeros((self.T, self.K - 1))
#         mu_0 = jax.random.multivariate_normal(mu_key, self.state.nu, self.state.Phi)
#         mu_t = mu_t.at[0].set(mu_0)
        
#         def sample_mu(carry, key_t):
#             mu_prev = carry
#             mu_next = jax.random.multivariate_normal(key_t, mu_prev, self.state.Phi)
#             return mu_next, mu_next
        
#         _, mu_scanned = jax.lax.scan(sample_mu, mu_0, jax.random.split(mu_key, self.T - 1))
#         mu_t = mu_t.at[1:].set(mu_scanned)

#         # 2. Generate gamma_tilde from mu
#         gamma_tilde = vmap(jax.random.multivariate_normal, in_axes=(None, 0, 0, None))(gamma_key, mu_t, self.state.Sigma, (self.N,))
        
#         # 3. Generate graph
#         gamma_expanded = vmap(_expand_gamma, in_axes=(0, None))(gamma_tilde, self.N)
#         pis = softmax(gamma_expanded, axis=-1)

#         def sample_E_t(key_t, pis_t, B):
#             z_ij = jax.random.categorical(key_t, jnp.log(pis_t), axis=-1, shape=(self.N, self.N))
#             z_ji = jax.random.categorical(key_t, jnp.log(pis_t), axis=-1, shape=(self.N, self.N))
#             probs = B[z_ij, z_ji]
#             return jax.random.bernoulli(key_t, probs)

#         E = vmap(sample_E_t, in_axes=(0, 0, None))(jax.random.split(e_key, self.T), pis, self.state.B)
#         self.state = self.state.replace(key=key, mu=mu_t, gamma_tilde=gamma_tilde) # Save generated values
#         return E.astype(jnp.int32)

#     # Accessor properties
#     @property
#     def B(self): return self.state.B
#     @property
#     def mu(self): return self.state.mu
#     @property
#     def Sigma(self): return self.state.Sigma
#     @property
#     def gamma_tilde(self): return self.state.gamma_tilde
#     @property
#     def Sigma_tilde(self): return self.state.Sigma_tilde
#     @property
#     def nu(self): return self.state.nu
#     @property
#     def Phi(self): return self.state.Phi
#     @property
#     def key(self): return self.state.key
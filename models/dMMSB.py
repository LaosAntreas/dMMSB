import jax
import jax.numpy as jnp
from jax.nn import softmax

from jax import vmap, jit
from jax.tree_util import register_pytree_node_class

EPS = 1e-10

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
    P = jnp.tile(jnp.eye(K - 1)[None, :, :], (T, 1, 1)) * 20 #shape (T, K-1, K-1) | NOTE: set large initial covariance
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
    jnp.log1p(-B_reshaped + EPS)) # log(1-B) more stable

    delta_exp_term = gamma_sum + log_bernoulli # shape (N,N,K,K)
    delta = softmax(delta_exp_term, axis=(-1,-2)) # shape (N,N,K,K)
    return delta # shape (N,N,K,K)

def _compute_g_H(gamma_hat, K):
    '''
    Compute g and H at gamma_hat
    gamma_hat: (N, K)
    Returns: g: (N, K), H: (N, K, K)
    '''
    g = softmax(gamma_hat, axis=-1) # shape (N,K)
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

        # Joseph form for numerical stability (P_t = P_pred_t - K_t @ P_pred_t)
        I = jnp.eye(P_pred_t.shape[0])
        term1 = (I - K_t) @ P_pred_t @ (I - K_t).T
        term2 = K_t @ (Sigma_t / N) @ K_t.T
        P_t = term1 + term2

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
    Update B. eq (26)
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
    return mu[0] 

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
    """
    Run the inner loop of the E-step until convergence.
    state: dMMSB_State
    E: adjacency matrix (T,N,N)
    max_inner_iters: maximum iterations for the inner loop
    tol: convergence tolerance
    """
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
    log1mB = jnp.log1p(- B_reshaped + EPS) # shape (1,1,1,K,K)

    #ll_matrix = delta * (E_reshaped * logB + (1.0 - E_reshaped) * log1mB) # shape (T,N,N,K,K)
    ll_matrix = jnp.where(E_reshaped == 0, log1mB * delta, delta * logB) # shape (T,N,N,K,K)
    ll = jnp.sum(ll_matrix)
    return ll

# ------------------------------

def find_best_initialization(state: dMMSB_State, E, trials=5):
    '''
    Run multiple trials of the inner loop with different random initializations and return the best state.
    state: dMMSB_State
    E: adjacency matrix (T,N,N)
    trials: number of random initializations
    Returns: best_state: dMMSB_State, best_ll: scalar
    '''
    def single_trial(state, E):
        state, inner_ll = inner_loop(state, E, 100, tol=1e-6)
        return state, inner_ll
    
    #init states with different keys
    keys = jax.random.split(state.key, trials)
    states = vmap(lambda k: state.replace(key=k))(keys)
    vmap_trial = vmap(single_trial, in_axes=(0, None), out_axes=(0,0))(states, E)
    states, lls = vmap_trial
    best_idx = jnp.argmax(lls)
    
    # Index into the PyTree to get the best state
    best_state = jax.tree_util.tree_map(lambda x: x[best_idx], states)
    return best_state, lls[best_idx]

# --- Public API ---

class jitdMMSB:
    """Implementation of the dMMSB model."""

    def __init__(self, nodes, roles, timesteps, **kwargs):
        self.N = nodes
        self.K = roles
        self.T = timesteps
        key_seed = kwargs.pop('key', 0)
        self.state = init_dMMSB_state(self.N, self.K, self.T, key=key_seed, **kwargs)

    def fit(self, E, max_inner_iters=100, max_outer_iters=100, tol=1e-6, verbose=False):
        '''
        Fit the dMMSB model to the data E using variational EM.
        E: adjacency matrix (T,N,N)
        max_inner_iters: maximum iterations for the inner loop
        max_outer_iters: maximum iterations for the outer loop
        tol: convergence tolerance
        verbose: whether to print progress of outer loop
        Returns: final log likelihood
        '''
        assert E.shape == (self.T, self.N, self.N), f"Expected E shape {(self.T, self.N, self.N)}, got {E.shape}"
        assert jnp.all(jnp.isin(E, jnp.array([0, 1]))), "E must be a binary adjacency matrix."
        i = 0 
        d_ll = jnp.inf
        prev_outer_ll = -jnp.inf

        # Multiple random initializations to avoid poor local minima
        initial_state, outer_ll = find_best_initialization(self.state, E, trials=5)
        self.state = initial_state

        # Outer Loop
        while(d_ll > tol and i < max_outer_iters):
            # Inner Loop (E-Step)
            self.state, inner_ll = inner_loop(self.state, E, max_inner_iters, tol)

            # M-Step
            self.state = m_step_outer_updates(self.state)

            # Convergence check
            i += 1
            outer_ll = inner_ll
            d_ll = jnp.abs(outer_ll - prev_outer_ll)
            prev_outer_ll = outer_ll
            if verbose:
                msg = f"[outer {i}/{max_outer_iters}] ll: {outer_ll:.4f}, d_ll: {d_ll:.6f}"
                print(msg, end="\r", flush=True)    
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
        
            key, subkey = jax.random.split(key)
            mu_0 = jax.random.multivariate_normal(subkey, self.nu, self.Phi)

            def sample_mu_step(carry, key_t):
                mu_prev = carry
                mu_t = jax.random.multivariate_normal(key_t, mu_prev, self.Phi)
                return mu_t, mu_t

            keys = jax.random.split(key, self.T - 1)
            _, mu_rest = jax.lax.scan(sample_mu_step, mu_0, keys)
            mu = jnp.concatenate([mu_0[None, :], mu_rest], axis=0)
            
            # 2. Generate gamma_tilde using vmap
            def sample_gamma_t(key_t, mu_t, Sigma_t):
                return jax.random.multivariate_normal(key_t, mu_t, Sigma_t, shape=(self.N,))

            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, self.T)
            gamma_tilde = vmap(sample_gamma_t)(keys, self.mu, self.Sigma)

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
            
            key, subkey = jax.random.split(key)
            mu_0 = jax.random.multivariate_normal(subkey, self.nu, self.Phi)

            def sample_mu_step(carry, key_t):
                mu_prev = carry
                mu_t = jax.random.multivariate_normal(key_t, mu_prev, self.Phi)
                return mu_t, mu_t

            keys = jax.random.split(key, self.T - 1)
            _, mu_rest = jax.lax.scan(sample_mu_step, mu_0, keys)
            mu = jnp.concatenate([mu_0[None, :], mu_rest], axis=0)
            
            def sample_gamma_t(key_t, mu_t, Sigma_t):
                return jax.random.multivariate_normal(key_t, mu_t, Sigma_t, shape=(self.N,))

            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, self.T)
            gamma_tilde = vmap(sample_gamma_t)(keys, self.mu, self.Sigma)
            self.state = self.state.replace(gamma_tilde=gamma_tilde, mu=mu, key=key)
        
        gamma_expanded = vmap(_expand_gamma, in_axes=(0, None))(self.gamma_tilde, self.state.N) # shape (T,N,K)
        pis = softmax(gamma_expanded, axis=-1) # shape (T,N,K)

        def sample_E(key, pis_t):
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            z_ij = jax.random.multinomial(subkey1, 1, pis_t, shape=(self.state.N, self.state.N, self.state.K)) # shape (N,N,K)
            z_ji = jax.random.multinomial(subkey2, 1, pis_t, shape=(self.state.N, self.state.N, self.state.K)) # shape (N,N,K)

            p = jnp.einsum('ijk, kl -> ijl', z_ij, self.state.B) # shape (N,N,K)
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

    

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

EPS = 1e-10

@register_pytree_node_class
class LNMMSB_State:
    """A PyTree class to hold the state of the LNMMSB model."""
    def __init__(self, N, K, key=None, B=None, mu=None, Sigma=None, gamma_tilde=None, Sigma_tilde=None, delta=None):
        # --- Static Configuration ---
        self.N = N
        self.K = K

        # --- Dynamic Parameters (JAX arrays) ---
        self.key = key 
        self.B = B
        self.mu = mu
        self.Sigma = Sigma
        self.gamma_tilde = gamma_tilde
        self.Sigma_tilde = Sigma_tilde
        self.delta = delta

    def tree_flatten(self):
        """Tells JAX how to flatten the object."""
        # The dynamic children are the JAX arrays that will be traced.
        children = (self.key, self.B, self.mu, self.Sigma, self.gamma_tilde, self.Sigma_tilde, self.delta) # FIX: Added key to children
        
        # The auxiliary data is static and won't be traced.
        aux_data = {'N': self.N, 'K': self.K}
        
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Tells JAX how to unflatten the object."""
        key, B, mu, Sigma, gamma_tilde, Sigma_tilde, delta = children
        return cls(N=aux_data['N'], K=aux_data['K'], key=key, B=B, mu=mu, Sigma=Sigma, 
                   gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde, delta=delta)

    def replace(self, **kwargs):
        """A helper method for immutably updating the state."""
        # FIX: Create a dictionary from the current state's attributes
        current_fields = {
            'key': self.key, 'B': self.B, 'mu': self.mu, 'Sigma': self.Sigma,
            'gamma_tilde': self.gamma_tilde, 'Sigma_tilde': self.Sigma_tilde, 'delta': self.delta
        }
        updated_fields = {**current_fields, **kwargs}
        
        return LNMMSB_State(N=self.N, K=self.K, **updated_fields)


def init_LNMMSB_state(N, K, key=0, B=None, mu=None, Sigma=None):
    """Initializes the LNMMSB state with given or random parameters."""
    rng = jax.random.PRNGKey(key)
    rng, b_key, mu_key, gamma_key = jax.random.split(rng, 4)

    if B is None:
        B = jax.random.uniform(b_key, (K, K))

    if mu is None:
        mu = jax.random.normal(mu_key, (K - 1,))
    if Sigma is None:
        Sigma = jnp.eye(K - 1) * 10
    
    # Variational parameters are initialized in the fit loop, so set to None here.
    gamma_tilde = jnp.zeros((N, K - 1))
    Sigma_tilde = jnp.zeros((N, K - 1, K - 1))
    delta = jnp.zeros((N, N, K, K))
    
    return LNMMSB_State(N=N, K=K, key=rng, B=B, mu=mu, Sigma=Sigma, gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde, delta=delta)


# --- Inference Functions ---

def _expand_gamma(gamma_km1, N): # FIX: Corrected argument name from Ν to N
    '''
    Help function to expand gamma from K-1 to K by appending zeros
    gamma_km1: (N, K-1) 
    returns: (N, K)
    '''
    zeros = jnp.zeros((N, 1))
    return jnp.concatenate([gamma_km1, zeros], axis=-1)


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


def compute_g_H(gamma_hat, K):
    '''
    Compute g and H at the K-dimensional expansion point gamma_hat.
    gamma_hat: (N, K)
    Returns: g: (N, K), H: (N, K, K)
    '''
    g = softmax(gamma_hat, axis=-1) # shape (N,K)
    H = jnp.einsum('ni,ij->nij', g, jnp.eye(K)) - jnp.einsum('ni,nj->nij', g, g)
    
    return g, H
    
def update_sigma_tilde(Sigma_inv, H, N, K):
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

def update_gamma_tilde(mu, Sigma_tilde, gamma_hat, delta, g, H, N, K):
    '''
    Update gamma_tilde using Laplace approximation.
    '''
    factor = 2.0 * N - 2.0
    
    m_expect = compute_m_expect(delta)
    
    mu_expanded = jnp.append(mu, 0.0)
    
    term_1_full = (m_expect - factor * g + 
                factor * jnp.einsum('nij,nj->ni', H, gamma_hat) - 
                factor * jnp.einsum('nij,j->ni', H, mu_expanded))

    term_1_km1 = term_1_full[:, :K - 1]

    gamma_tilde = mu[None, :] + jnp.einsum('nij,nj->ni', Sigma_tilde, term_1_km1)

    return gamma_tilde

def compute_m_expect(delta):
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

    return m_expect

def update_B(E, delta):
    '''
    Update B. eq (24)
    E: adjacency matrix (N,N)
    delta: (N,N,K,K)
    Returns: B_new: (K,K)
    '''
    extended_E = E[:, :, None, None]

    num = jnp.sum(extended_E * delta, axis=(0,1))
    den = jnp.sum(delta, axis=(0,1))

    B_new = jnp.where(den < EPS, 
                0.5 * jnp.ones_like(num),  
                num / jnp.maximum(den, EPS))
    B_new = jnp.clip(B_new, 1e-6, 1 - 1e-6)

    return B_new

def update_mu_sigma(gamma_tilde, Sigma_tilde):
    '''
    Update mu and Sigma. eq (13)
    '''
    mu = jnp.mean(gamma_tilde, axis=0)

    avg_sigma_tilde = jnp.mean(Sigma_tilde, axis=0)
    cov_gamma_tilde = jnp.cov(gamma_tilde, rowvar=False) 
    cov_gamma_tilde += 1e-8 * jnp.eye(cov_gamma_tilde.shape[0])
    
    updated_Sigma = avg_sigma_tilde + cov_gamma_tilde

    return mu, updated_Sigma


def find_best_initialization(state: LNMMSB_State, E, trials=5):
    '''
    Run multiple trials of the inner loop with different random initializations and return the best state.
    state: LMMSB_State
    E: adjacency matrix (N,N)
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


# --- Jitted Step Functions ---

def e_step_update(state, Sigma_inv, E):
    '''
    Perform one E-step update: update delta, gamma_tilde, Sigma_tilde, B
    '''
    gamma_hat = _expand_gamma(state.gamma_tilde, state.N)

    delta = _compute_deltas(gamma_hat, state.B, E)

    g, H = compute_g_H(gamma_hat, state.K)
    Sigma_tilde = update_sigma_tilde(Sigma_inv, H, state.N, state.K)
    gamma_tilde = update_gamma_tilde(state.mu, Sigma_tilde, gamma_hat, delta, g, H, state.N, state.K)

    B = update_B(E, delta)

    new_state = state.replace(delta=delta, gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde, B=B)
    
    return new_state

@jit
def inner_loop(state: LNMMSB_State, E, max_inner_iters=100, tol=1e-6):
    '''
    Run the inner loop of the E-step until convergence.
    state: LNMMSB_State
    E: adjacency matrix (N,N)
    max_inner_iters: maximum iterations for the inner loop
    tol: convergence tolerance
    '''
    j = 0 
    d_ll = jnp.inf
    prev_inner_ll = -jnp.inf

    key, subkey = jax.random.split(state.key)
    gamma_tilde_init = jax.random.multivariate_normal(subkey, state.mu, state.Sigma, (state.N,))
    state = state.replace(key=key, gamma_tilde=gamma_tilde_init)
    
    jitter = 1e-6 * jnp.eye(state.K - 1)
    Sigma_inv = jnp.linalg.inv(state.Sigma + jitter)

    def cond_fun(carry):
        state, j, inner_d_ll, prev_inner_ll = carry
        return (inner_d_ll > tol) & (j < max_inner_iters)
    
    def body_fun(carry):
        state, j, inner_d_ll, prev_inner_ll = carry

        state = e_step_update(state, Sigma_inv, E)

        j += 1
        inner_ll = log_likelihood(state.delta, state.B, E) 
        inner_d_ll = jnp.abs(inner_ll - prev_inner_ll)
        prev_inner_ll = inner_ll

        return (state, j, inner_d_ll, prev_inner_ll)

    init_carry = (state, j, d_ll, prev_inner_ll)
    state, j, d_ll, prev_ll = jax.lax.while_loop(cond_fun, body_fun, init_carry)

    return state, prev_ll

@jit
def m_step_update(state):
    '''
    Perform one M-step update: update mu and Sigma
    '''
    mu, Sigma = update_mu_sigma(state.gamma_tilde, state.Sigma_tilde)
    
    new_state = state.replace(mu=mu, Sigma=Sigma)
    
    return new_state

@jit
def log_likelihood(delta, B,  E):
    '''
    Compute log likelihood of data E given current parameters and deltas.
    '''
    E_reshaped = E[:, :, None, None]
    B_reshaped = B[None, None, :, :]
    B_reshaped = jnp.clip(B_reshaped, EPS, 1 - EPS)
    logB = jnp.log(B_reshaped)
    log1mB = jnp.log1p(- B_reshaped)

    #ll_matrix = E_reshaped * logB + (1 - E_reshaped) * log1mB
    ll_matrix = jnp.where(E_reshaped == 1, logB, log1mB)
    ll = jnp.sum(delta * ll_matrix)

    return ll

# ------------------------------

def find_best_initialization(state: LNMMSB_State, E, trials=5):
    '''
    Run multiple trials of the inner loop with different random initializations and return the best state.
    state: LMMSB_State
    E: adjacency matrix (N,N)
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

class jitLNMMSB:
    """A JAX-accelerated implementation of the LNMMSB model."""
    
    def __init__(self, nodes, roles, **kwargs):
        self.N = nodes
        self.K = roles
        key_seed = kwargs.pop('key', 0)
        self.state = init_LNMMSB_state(nodes, roles, key_seed, **kwargs)

    def fit(self, E, max_inner_iters=100, max_outer_iters=100, tol=1e-6, verbose=False):
        '''
        Fit the model to adjacency matrix E using variational EM
        E: adjacency matrix (N,N)
        max_inner_iters: maximum iterations for the inner loop
        max_outer_iters: maximum iterations for the outer loop
        tol: convergence tolerance
        verbose: whether to print progress
        Returns: final log likelihood
        '''
        assert E.shape == (self.N, self.N), f"Expected E shape {(self.N, self.N)}, got {E.shape}"
        assert jnp.all(jnp.isin(E, jnp.array([0, 1]))), "E must be a binary adjacency matrix."

        i = 0 
        d_ll = jnp.inf
        prev_outer_ll = -jnp.inf

        # Multiple random initializations to avoid poor local minima
        initial_state, outer_ll = find_best_initialization(self.state, E, trials=5)
        self.state = initial_state
        
        while(d_ll > tol and i < max_outer_iters):

            key, subkey = jax.random.split(self.state.key)
            gamma_tilde_init = jax.random.multivariate_normal(subkey, self.state.mu, self.state.Sigma, (self.N,))
            self.state = self.state.replace(key=key, gamma_tilde=gamma_tilde_init)
            
            jitter = 1e-6 * jnp.eye(self.K - 1)
            Sigma_inv = jnp.linalg.inv(self.state.Sigma + jitter)

            #Inner loop (E-step)
            self.state, inner_ll = inner_loop(self.state, E, max_inner_iters, tol)

            #M-step
            self.state = m_step_update(self.state)

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
        Generate a graph from the learned model parameters
        Returns: adjacency matrix (N,N)
        '''
        key, subkey_1, subkey_2, subkey_3, subkey_4 = jax.random.split(self.state.key, 5)
        if jnp.all(self.state.gamma_tilde == jnp.zeros_like(self.state.gamma_tilde)):
            assert self.state.mu is not None and self.state.Sigma is not None, "Must initialize mu and Sigma before generating a graph if gamma_tilde is not set."

            gamma_tilde= jax.random.multivariate_normal(subkey_1, self.state.mu, self.state.Sigma, (self.state.N,)) # shape (N,K)
            self.state = self.state.replace(key=key, gamma_tilde=gamma_tilde)
            
            if(self.gamma_tilde.shape[1] == self.state.K - 1):
                gamma_tilde = _expand_gamma(self.state.gamma_tilde, self.state.N)
        else:
            if(self.state.gamma_tilde.shape[1] == self.state.K - 1):
                gamma_tilde = _expand_gamma(self.state.gamma_tilde, self.state.N)
        
        pis = softmax(gamma_tilde, axis=-1)

        z_ij = jax.random.multinomial(subkey_2, 1, pis, shape=(self.state.N,self.state.N,self.state.K)) # shape (N,N,K) (sender)
        z_ji = jax.random.multinomial(subkey_3, 1, pis, shape=(self.state.N,self.state.N,self.state.K)) # shape (N,N,K) (receiver)

        
        p = jnp.einsum('ijk, kl -> ijl', z_ij, self.state.B) # shape (N,N,K)
        p = jnp.einsum('ijl, jil -> ij', p, z_ji) # shape (N,N)

        E_sampled = jax.random.bernoulli(subkey_4, p) # shape (N,N)
        return E_sampled
        
    
    # accessor methods
    @property
    def B(self): return self.state.B

    @property
    def mu(self): return self.state.mu

    @property
    def Sigma(self): return self.state.Sigma

    @property
    def gamma_tilde(self): return self.state.gamma_tilde



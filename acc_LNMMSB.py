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
    gamma_tilde = None
    Sigma_tilde = None
    
    return LNMMSB_State(N=N, K=K, key=rng, B=B, mu=mu, Sigma=Sigma, gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde, delta=None)


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
    gamma_tilde: (N,K)
    B: (K,K)
    E: adjacency matrix (N,N)
    Returns: delta (N,N,K,K)
    '''
    gamma_i = gamma_tilde[:, None, :, None]
    gamma_j = gamma_tilde[None, :, None, :]

    gamma_sum = gamma_i + gamma_j

    B_reshaped = B[None, None, :, :]
    E_reshaped = E[:, :, None, None]

    log_bernoulli = jnp.where(E_reshaped == 1,
                            jnp.log(B_reshaped + EPS),
                            jnp.log1p(-B_reshaped + EPS))

    delta_exp_term = gamma_sum + log_bernoulli
    max_delta_exp = jnp.max(delta_exp_term, axis=(-1,-2), keepdims=True)
    delta = jnp.exp(delta_exp_term - (max_delta_exp + logsumexp(delta_exp_term - max_delta_exp, axis=(-1,-2), keepdims=True)))
    return delta

def compute_g_H(gamma_hat, K):
    '''
    Compute g and H at the K-dimensional expansion point gamma_hat.
    gamma_hat: (N, K)
    Returns: g: (N, K), H: (N, K, K)
    '''
    max_gamma = jnp.max(gamma_hat, axis=-1, keepdims=True)
    g = jnp.exp(gamma_hat - (max_gamma + logsumexp(gamma_hat - max_gamma, axis=-1, keepdims=True)))
    H = jnp.einsum('ni,ij->nij', g, jnp.eye(K)) - jnp.einsum('ni,nj->nij', g, g)
    
    return g, H
    
def update_sigma_tilde(Sigma_inv, H, N, K):
    '''
    Compute Sigma_tilde = (Sigma^{-1} + (2N-2) H_{1:K-1, 1:K-1})^{-1}
    Sigma_inv: (K-1, K-1)
    H: (N, K, K) Hessian from the K-dimensional expansion
    Returns: Sigma_tilde: (N, K-1, K-1)
    '''
    H_km1 = H[:, :K - 1, :K - 1]
    
    factor = 2.0 * N - 2.0
    A = Sigma_inv[None, :, :] + factor * H_km1
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
    Update B using eq (24): β_{k,l} = sum_{i,j} eij δij(k,l) / sum_{i,j} δij(k,l)
    '''
    extended_E = E[:, :, None, None]

    num = jnp.sum(extended_E * delta, axis=(0,1))
    den = jnp.sum(delta, axis=(0,1))

    B_new = jnp.where(den < EPS, 
                0.5 * jnp.ones_like(num),  
                num / jnp.maximum(den, EPS))
    B_new = jnp.clip(B_new, EPS, 1 - EPS)

    return B_new

def update_mu_sigma(gamma_tilde, Sigma_tilde):
    '''
    Update mu and Sigma based on eq (13)
    '''
    mu = jnp.mean(gamma_tilde, axis=0)

    avg_sigma_tilde = jnp.mean(Sigma_tilde, axis=0)
    cov_gamma_tilde = jnp.cov(gamma_tilde, rowvar=False)
    cov_gamma_tilde += 1e-8 * jnp.eye(cov_gamma_tilde.shape[0])
    
    updated_Sigma = avg_sigma_tilde + cov_gamma_tilde

    return mu, updated_Sigma

@partial(jit, static_argnames=['N', 'K'])
def e_step_update(state, Sigma_inv, E, N, K):
    '''
    Perform one E-step update: update delta, gamma_tilde, Sigma_tilde, B
    '''
    gamma_hat = _expand_gamma(state.gamma_tilde, N)

    delta = _compute_deltas(gamma_hat, state.B, E)

    g, H = compute_g_H(gamma_hat, K)
    Sigma_tilde = update_sigma_tilde(Sigma_inv, H, N, K)
    gamma_tilde = update_gamma_tilde(state.mu, Sigma_tilde, gamma_hat, delta, g, H, N, K)

    B = update_B(E, delta)

    new_state = state.replace(delta=delta, gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde, B=B)
    
    return new_state

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

    logB = jnp.log(B_reshaped + EPS)
    log1mB = jnp.log(1 - B_reshaped + EPS)

    ll_matrix = E_reshaped * logB + (1 - E_reshaped) * log1mB
    ll = jnp.sum(delta * ll_matrix)

    return ll


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
        '''
        # FIX: Compile the e_step_update function once
        e_step_jitted = partial(e_step_update, N=self.N, K=self.K)

        i = 0 
        d_ll = jnp.inf
        prev_outer_ll = -jnp.inf
        
        while(d_ll > tol and i < max_outer_iters):
            if verbose:
                print(f"[outer {i}] mu: {self.mu}, Sigma diag: {jnp.diag(self.Sigma)}, B: {self.B}")

            # FIX: Correctly manage state and PRNG keys
            key, subkey = jax.random.split(self.state.key)
            gamma_tilde_init = jax.random.multivariate_normal(subkey, self.state.mu, self.state.Sigma, (self.N,))
            self.state = self.state.replace(key=key, gamma_tilde=gamma_tilde_init)
            
            jitter = 1e-6 * jnp.eye(self.K - 1)
            Sigma_inv = jnp.linalg.inv(self.state.Sigma + jitter)

            j = 0
            inner_d_ll = jnp.inf
            prev_inner_ll = -jnp.inf
            while(inner_d_ll > tol and j < max_inner_iters):

                self.state = e_step_jitted(self.state, Sigma_inv, E)
                
                j += 1
                inner_ll = log_likelihood(self.state.delta, self.state.B, E) 
                inner_d_ll = jnp.abs(inner_ll - prev_inner_ll)
                prev_inner_ll = inner_ll
                if verbose:
                    print(f"  [inner {j}] ll: {inner_ll:.4f}, d_ll: {inner_d_ll:.6f}")
            
            self.state = m_step_update(self.state)

            i += 1
            outer_ll = inner_ll
            d_ll = jnp.abs(outer_ll - prev_outer_ll)
            prev_outer_ll = outer_ll
            if verbose:
                print(f"[outer {i}] ll: {outer_ll:.4f}, d_ll: {d_ll:.6f}")
                print(10 * "-", "Params", 10 * "-")
                print(f"mu: {self.mu}")
                print(f"Sigma: {self.Sigma}")
                print(f"B: {self.B}")
                print(30 * "-")

        return outer_ll

    def generate_graph(self):
        '''
        Generate a graph from the learned model parameters
        Returns: adjacency matrix (N,N)
        '''
        key, subkey_1, subkey_2, subkey_3, subkey_4 = jax.random.split(self.state.key, 5)
        if self.gamma_tilde is None:
            print("Sampling new gammas")
            self.gamma_tilde= jax.random.multivariate_normal(subkey_1, self.state.mu, self.state.Sigma, (self.state.N,)) # shape (N,K)
            if(self.gamma_tilde.shape[1] == self.state.K - 1):
                gamma_tilde = _expand_gamma(self.state.gamma_tilde, self.state.N)
            pis = softmax(self.state.gamma_tilde, axis=-1) # shape (N,K)
        else:
            if(self.state.gamma_tilde.shape[1] == self.state.K - 1):
                gamma_tilde = _expand_gamma(self.state.gamma_tilde, self.state.N)
            pis = softmax(gamma_tilde, axis=-1)

        #print("pis shape", pis)
        z_ij = jax.random.multinomial(subkey_2, 1, pis, shape=(self.state.N,self.state.N,self.state.K)) # shape (N,N,K)
        #print("z_ij shape", z_ij.shape)
        z_ji = jax.random.multinomial(subkey_3, 1, pis, shape=(self.state.N,self.state.N,self.state.K)) # shape (N,N,K)
        #print("z_ji shape", z_ji.shape)
        
        p = jnp.einsum('ijk, kl -> ijl', z_ij, self.state.B) # shape (N,N,K)
        #print("p first einsum", p.shape)
        p = jnp.einsum('ijl, jil -> ij', p, z_ji) # shape (N,N)
        #print("p second einsum", p.shape)

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



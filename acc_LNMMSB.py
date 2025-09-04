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

EPS = 1e-6

@register_pytree_node_class
class LNMMSB_State:
    """A PyTree class to hold the state of the LNMMSB model."""
    def __init__(self, N, K, B=None, mu=None, Sigma=None, gamma_tilde=None, Sigma_tilde=None, delta=None):
        # --- Static Configuration ---
        self.N = N
        self.K = K

        # --- Dynamic Parameters (JAX arrays) ---
        self.B = B
        self.mu = mu
        self.Sigma = Sigma
        self.gamma_tilde = gamma_tilde
        self.Sigma_tilde = Sigma_tilde
        self.delta = delta

    def tree_flatten(self):
        """Tells JAX how to flatten the object."""
        # The dynamic children are the JAX arrays that will be traced.
        children = (self.B, self.mu, self.Sigma, self.gamma_tilde, self.Sigma_tilde, self.delta)
        
        # The auxiliary data is static and won't be traced.
        aux_data = {'N': self.N, 'K': self.K}
        
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Tells JAX how to unflatten the object."""
        B, mu, Sigma, gamma_tilde, Sigma_tilde, delta = children
        return cls(N=aux_data['N'], K=aux_data['K'], B=B, mu=mu, Sigma=Sigma, 
                   gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde, delta=delta)

    def replace(self, **kwargs):
        """A helper method for immutably updating the state."""
        current_fields = {
            'B': self.B, 'mu': self.mu, 'Sigma': self.Sigma,
            'gamma_tilde': self.gamma_tilde, 'Sigma_tilde': self.Sigma_tilde, 'delta': self.delta
        }
        updated_fields = {**current_fields, **kwargs}
        
        return LNMMSB_State(N=self.N, K=self.K, **updated_fields)


def _init_LNMMSB_state(N, K, key=0, B=None, mu=None, Sigma=None, gamma_tilde=None, Sigma_tilde=None):
    """Initializes the LNMMSB state with given or random parameters."""
    rng = jax.random.PRNGKey(key)

    if B is None:
        B = jax.random.uniform(rng, (K, K))  # shape (K,K)
    if mu is None:
        mu = jax.random.multivariate_normal(rng, jnp.zeros(K), jnp.eye(K))  # shape (K,)
    if Sigma is None:
        Sigma = jnp.eye(K) * 10  # shape (K,K)
    if gamma_tilde is None:
        gamma_tilde = jax.random.multivariate_normal(rng, mu, Sigma, (N,))  # shape (N,K)
    if Sigma_tilde is None:
        Sigma_tilde = jnp.tile(Sigma[None, :, :], (N, 1, 1))  # shape (N,K,K)   
    
    return LNMMSB_State(N=N, K=K, B=B, mu=mu, Sigma=Sigma)


@jit
def _compute_deltas(state : LNMMSB_State, E):
    '''
    Compute delta matrices for all pairs (i,j) given current gamma_tilde and B
    B: (K,K)
    gamma_tilde: (N,K)
    E: adjacency matrix (N,N)
    Returns: delta (N,N,K,K)
    '''
    
    gamma_i = state.gamma_tilde[:, None, :, None] # shape (N,1,K,1)
    gamma_j = state.gamma_tilde[None, :, None, :] # shape (1,N,1,K)

    gamma_sum = gamma_i + gamma_j # shape (N,N,K,K)

    #print("gamma sum is finite:", jnp.isfinite(gamma_sum).all())
    B_reshaped = state.B[None, None, :, :] # shape (1,1,K,K)
    E_reshaped = E[:, :, None, None] # shape (N,N,1,1)

    bernoulli_term = jnp.where(E_reshaped == 1, B_reshaped, 1 - B_reshaped) # shape (N,N,K,K)

    delta_exp_term = gamma_sum + jnp.log(bernoulli_term + EPS) # shape (N,N,K,K)
    #print("delta exp term is finite:", jnp.isfinite(delta_exp_term).all())
    max_delta_exp = jnp.max(delta_exp_term, axis=(-1,-2), keepdims=True)
    delta = jnp.exp(delta_exp_term - (max_delta_exp + logsumexp(delta_exp_term - max_delta_exp, axis=(-1,-2), keepdims=True))) # shape (N,N,K,K) logsumexp trick for numerical stability
    return delta # shape (N,N,K,K)

@jit
def _log_likelihood(state : LNMMSB_State, E):
    '''
    Compute log likelihood of data E given current parameters and deltas. Based on eqn (23).
    delta: (N,N,K,K)
    B: (K,K)
    E: adjacency matrix (N,N)
    Returns: scalar log likelihood
    '''
    E_reshaped = E[:, :, None, None] # shape (N,N,1,1)
    B_reshaped = state.B[None, None, :, :] # shape (1,1,K,K)

    logB = jnp.log(B_reshaped + EPS) # shape (1,1,K,K)
    log1mB = jnp.log(1 - B_reshaped + EPS) # shape (1,1,K,K)

    ll_matrix = E_reshaped * logB + (1 - E_reshaped) * log1mB # shape (N,N,K,K)
    ll = jnp.sum(state.delta * ll_matrix) # scalar

    return ll

@partial(jit, static_argnames=['K',])
def _compute_g_H(gamma_hat, K):
    '''
    Compute g and H at gamma_hat
    gamma_hat: (N, K)
    Returns: g: (N, K), H: (N, K, K)
    '''
    #g = jnp.exp(gamma_hat) / jnp.sum(jnp.exp(gamma_hat), axis=-1, keepdims=True) # shape (N,K)
    max_gamma = jnp.max(gamma_hat, axis=-1, keepdims=True)
    g = jnp.exp(gamma_hat - (max_gamma + logsumexp(gamma_hat - max_gamma, axis=-1, keepdims=True))) # shape (N,K) logsumexp trick for numerical stability
    H = jnp.einsum('ni,ij->nij', g, jnp.eye(K)) - jnp.einsum('ni,nj->nij', g, g) # shape (N,K,K)
    # print("g is finite:", jnp.isfinite(g).all())
    # print("H is finite:", jnp.isfinite(H).all())
    return g, H

@partial(jit, static_argnames=['N',])
def _update_sigma_tilde(Sigma_inv, H, N):
    '''
    Compute Sigma_tilde = (Sigma^{-1} + (2N-2) H)^{-1}
    Sigma_inv: (K,K)
    H: (N,K,K) Hessian at gamma_hat
    Returns: Sigma_tilde: (N,K,K)
    '''
    factor = 2.0 * N - 2.0
    A = Sigma_inv[None, :, :] + factor * H # shape (N,K,K)
    #jitter = 1e-6 * jnp.eye(self.K)
    #A = A + jitter[None, :, :]
    Sigma_tilde = jnp.linalg.inv(A)
    return Sigma_tilde # shape (N,K,K)

@jit
def _update_gamma_tilde(state: LNMMSB_State, g , H):
    '''
    Update gamma_tilde using Laplace approximation

    mu: (K,)
    Sigma_tilde: (N,K,K)
    gamma_hat: (N,K)
    m_expect: (N,K)
    Returns: gamma_tilde: (N,K), Sigma_tilde: (N,K,K)
    '''
    # g, H = self.compute_g_H(gamma_hat) # g: (N,K), H: (N,K,K)
    factor = 2.0 * state.N - 2.0 #scalar

    m_expect = _compute_m_expect(state.delta) # shape (N,K)

    term_1 = m_expect - factor * g + factor * jnp.einsum('nij,nj->ni', H, state.gamma_tilde) - factor * jnp.einsum('nij,j->ni', H, state.mu) # shape (N,K)

    gamma_tilde = state.mu[None, :] + jnp.einsum('nij,nj->ni', state.Sigma_tilde, term_1) # shape (N,K)

    return gamma_tilde # shape (N,K)

@jit
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

@jit
def _update_B(delta, E):
    '''
    Update B using eq (24): β_{k,l} = sum_{i,j} eij δij(k,l) / sum_{i,j} δij(k,l)
    E: adjacency matrix (N,N)
    delta: (N,N,K,K)
    Returns: updated B (K,K)
    '''
    extended_E = E[:, :, None, None] # shape (N,N,1,1)

    num = jnp.sum(extended_E * delta, axis=(0,1)) # shape (K,K)
    den = jnp.sum(delta, axis=(0,1)) # shape (K,K)

    B_new = (num + EPS) / (den + EPS) # shape (K,K)
    B_new = jnp.clip(B_new, 1e-6, 1 - 1e-6)

    return B_new # shape (K,K)


@jit
def _update_mu_sigma(gamma_tilde, Sigma_tilde):
    '''
    Update mu and Sigma based on eq (13)
    gamma_tilde: (N,K)
    Sigma_tilde: (N,K,K)
    Returns: updated mu (K,), updated Sigma (K,K)
    '''
    mu = jnp.mean(gamma_tilde, axis=0) # shape (K,)

    avg_sigma_tilde = jnp.mean(Sigma_tilde, axis=0)  # shape (K,K)
    cov_gamma_tilde = jnp.cov(gamma_tilde, rowvar=False) # shape (K,K)
    #print('cov shape:', cov_gamma_tilde)

    #print('cov gamma tilde is finite:', jnp.isfinite(cov_gamma_tilde).all())
    
    updated_Sigma = avg_sigma_tilde + cov_gamma_tilde # shape (K,K)
    Sigma = updated_Sigma

    return mu, Sigma



class jit_LNMMSB():
    def __init__(self, nodes, roles, **kwargs):
        self.N = nodes
        self.K = roles

        key = kwargs.get('key', 0)
        B = kwargs.get('B', None)
        mu = kwargs.get('mu', None)
        Sigma = kwargs.get('Sigma', None)
        gamma_tilde = kwargs.get('gamma_tilde', None)
        Sigma_tilde = kwargs.get('Sigma_tilde', None)

        if key is None:
            self.key = jax.random.PRNGKey(0)
        else:
            self.key = jax.random.PRNGKey(key)

        self.state = _init_LNMMSB_state(self.N, self.K, B=B, mu=mu, Sigma=Sigma, gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde)


    
    def fit(self, E, max_inner_iters=100, max_outer_iters=100, tol=1e-6, verbose=False):
        '''
        Fit the model to adjacency matrix E using variational EM
        Algorithm described in section 4.2 of the paper
        
        E: adjacency matrix (N,N)
        max_inner_iters: maximum iterations for inner loop
        max_outer_iters: maximum iterations for outer loop
        tol: tolerance for convergence
        verbose: whether to print progress
        '''
        
        i = 0 
        d_ll = jnp.inf
        prev_outer_ll = -jnp.inf
        gamma_tilde = jax.random.multivariate_normal(self.key, self.state.mu, self.state.Sigma, (self.state.N,)) # shape (N,K) #NOTE: temporary check
        self.state = self.state.replace(gamma_tilde=gamma_tilde)
        while(d_ll > tol and i < max_outer_iters): # 2 (outer loop)
            if verbose:
                print(f"[outer {i}] mu: {self.mu}, Sigma diag: {jnp.diag(self.Sigma)}, B: {self.B}")

            #initialize q(gamma) parameters
            
            g, H = _compute_g_H(self.state.gamma_tilde, self.state.K) # g: (N,K), H: (N,K,K)

            jitter = EPS * jnp.eye(self.K) # for numerical stability
            Sigma_inv = jnp.linalg.inv(self.state.Sigma + jitter) # shape (K,K)
            
            Sigma_tilde = _update_sigma_tilde(Sigma_inv, H, self.state.N) # shape (N,K,K)

            self.state = self.state.replace(gamma_tilde=gamma_tilde, Sigma_tilde=Sigma_tilde)

            #NOTE:add multiple runs with different initializaitons and use of VMAP
            j = 0
            inner_d_ll = jnp.inf
            prev_inner_ll = -jnp.inf
            while(inner_d_ll > tol and j < max_inner_iters): # 2.2 inner loop
                # 2.2.1 update q_z (delta)
                delta = _compute_deltas(self.state, E) # shape (N,N,K,K)
                self.state = self.state.replace(delta=delta)
                # 2.2.2 update q_gamma parameters
        
                g, H = _compute_g_H(self.state.gamma_tilde, self.state.K) # g: (N,K), H: (N,K,K)
                new_Sigma_tilde = _update_sigma_tilde(Sigma_inv, H, self.state.N) # shape (N,K,K)
                new_gamma_tilde = _update_gamma_tilde(self.state, g, H,) # shape (N,K)
                self.state = self.state.replace(gamma_tilde=new_gamma_tilde, Sigma_tilde=new_Sigma_tilde)
                # 2.2.3 update B
                new_B = _update_B(self.state.delta, E) # shape (K,K)

                self.state = self.state.replace(B=new_B)

                #convergence check
                j += 1
                inner_ll = _log_likelihood(self.state, E) 
                inner_d_ll = jnp.abs(inner_ll - prev_inner_ll)
                #print("inner ll and prev:", inner_ll, prev_inner_ll)    
                prev_inner_ll = inner_ll

                if verbose:
                    print(f"  [inner {j}] ll: {inner_ll:.4f}, d_ll: {inner_d_ll:.6f}")

            #print(50*'-')
            #print(f"Gammas: {self.gamma_tilde}")
            #print(50*'-')
            # 2.3 update mu, Sigma
            mu, Sigma = _update_mu_sigma(self.state.gamma_tilde, self.state.Sigma_tilde) # mu: (K,), Sigma: (K,K)

            self.state = self.state.replace(mu=mu, Sigma=Sigma)

            #convergence check
            i += 1
            outer_ll = inner_ll #last inner ll is outer ll
            d_ll = jnp.abs(outer_ll - prev_outer_ll)
            prev_outer_ll = outer_ll
    
    def generate_graph(self):
        '''
        Generate a graph from the learned model parameters
        Returns: adjacency matrix (N,N)
        '''
        # Sample gamma for each node
        if self.gamma_tilde is None:
            gammas = jax.random.multivariate_normal(self.key, self.mu, self.Sigma, (self.N,)) # shape (N,K)
            pis = softmax(gammas, axis=-1) # shape (N,K)
        else:
            pis = self.gamma_tilde

        #print("pis shape", pis)
        z_ij = jax.random.multinomial(self.key, 1, pis, shape=(self.N,self.N,self.K)) # shape (N,N,K)
        #print("z_ij shape", z_ij.shape)
        z_ji = jax.random.multinomial(self.key, 1, pis, shape=(self.N, self.N, self.K)) # shape (N,N,K)
        #print("z_ji shape", z_ji.shape)
        
        p = jnp.einsum('ijk, kl -> ijl', z_ij, self.B) # shape (N,N,K)
        #print("p first einsum", p.shape)
        p = jnp.einsum('ijl, jil -> ij', p, z_ji) # shape (N,N)
        #print("p second einsum", p.shape)

        E_sampled = jax.random.bernoulli(self.key, p) # shape (N,N)
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
   







                


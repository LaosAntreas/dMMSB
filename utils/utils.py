import jax
import jax.numpy as jnp
from itertools import permutations

def compute_best_permutation(true_roles, inferred_roles):
    best_perm = None 
    best_loss = jnp.inf
    for perm in permutations(range(true_roles.shape[-1])):
        loss = jnp.mean(jnp.abs(true_roles - inferred_roles[:, perm]))
        if loss < best_loss:
            best_loss = loss
            best_perm = perm
    return best_perm

def l1_loss(true, pred):
    return jnp.mean(jnp.abs(true - pred))
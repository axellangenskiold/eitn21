"""Compatibility shim for broken third-party ``commpy`` imports.

Some ``commpy`` distributions still try to import ``is_prime`` and
``modinv`` from a top-level ``CommPy`` module. Providing those names here
lets notebook imports such as ``from commpy.channelcoding import Trellis``
work without patching global site-packages.
"""


def is_prime(n: int) -> bool:
    """Return True when ``n`` is a prime number."""
    if n <= 1:
        return False
    return all(n % i != 0 for i in range(2, int(n**0.5) + 1))


def modinv(a: int, p: int) -> int:
    """Return the multiplicative inverse of ``a`` modulo ``p``."""
    t, new_t = 0, 1
    r, new_r = p, a

    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r

    if r > 1:
        raise ValueError(f"{a} is not invertible mod {p}")
    if t < 0:
        t += p
    return t

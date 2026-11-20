#!/usr/bin/env python
"""Generate the erfc coefficients of jqmc/rs/src/fastmath.rs (usage: python gen_erfc.py 26; needs mpmath)."""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Chebyshev fit of erfcx(z) = exp(z^2) erfc(z) as a polynomial in u = 2t-1, t = K/(K+z), z in [0, ZMAX]
import mpmath as mp, sys

mp.mp.dps = 60
K = mp.mpf(3)
ZMAX = mp.mpf(6.5)
DEG = int(sys.argv[1])
tmin = K / (K + ZMAX)


def f(u):
    t = (u + 1) / 2 * (1 - tmin) + tmin
    z = K / t - K
    return mp.erfc(z) * mp.exp(z * z)


n = DEG + 1
N = 200
c = [mp.mpf(0)] * n
nodes = [mp.cos(mp.pi * (k + mp.mpf(1) / 2) / N) for k in range(N)]
vals = [f(x) for x in nodes]
for j in range(n):
    c[j] = 2 / mp.mpf(N) * sum(vals[k] * mp.cos(mp.pi * j * (k + mp.mpf(1) / 2) / N) for k in range(N))
c[0] /= 2
# Chebyshev -> monomial in u
T = [[mp.mpf(1)], [mp.mpf(0), mp.mpf(1)]]
for k in range(2, n):
    p = [mp.mpf(0)] * (k + 1)
    for i, a in enumerate(T[k - 1]):
        p[i + 1] += 2 * a
    for i, a in enumerate(T[k - 2]):
        p[i] -= a
    T.append(p)
m = [mp.mpf(0)] * n
for k in range(n):
    for i, a in enumerate(T[k]):
        m[i] += c[k] * a
print("// tail cheb coeffs:", [mp.nstr(abs(x), 3) for x in c[-3:]])
print(f"pub const TMIN: f64 = {mp.nstr(tmin, 20)};\npub const K: f64 = 3.0;\npub const C: [f64; {n}] = [")
for x in m:
    print("    " + mp.nstr(x, 20) + ",")
print("];")

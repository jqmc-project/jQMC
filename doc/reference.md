# References

This page collects the original literature of the methods implemented in
jQMC. Please cite the relevant papers together with the jQMC papers listed in
{doc}`citation` when you publish results obtained with the corresponding
functionality.

## Wave function ansatz

(reference_jastrow)=

### Jastrow factor

The one-body, two-body and three/four-body Jastrow factors expanded in atomic
orbitals ({ref}`sec:Jastrow`) follow the form developed for the
Jastrow-correlated AGP wave function, and the cusp conditions they enforce
are Kato's:

- [T. Kato, Commun. Pure Appl. Math. 10, 151 (1957)](https://doi.org/10.1002/cpa.3160100201)
- [M. Casula and S. Sorella, J. Chem. Phys. 119, 6500 (2003)](https://doi.org/10.1063/1.1604379)
- [M. Casula, C. Attaccalite, and S. Sorella, J. Chem. Phys. 121, 7110 (2004)](https://doi.org/10.1063/1.1794632)

The smooth periodic distance used by the one- and two-body Jastrow factors in
periodic systems ({ref}`sec:periodic_jastrow`):

- [A. J. Williamson, G. Rajagopal, R. J. Needs, L. M. Fraser, W. M. C. Foulkes, Y. Wang, and M.-Y. Chou, Phys. Rev. B 55, R4851 (1997)](https://doi.org/10.1103/PhysRevB.55.R4851)

(reference_dtn)=

### Drummond--Towler--Needs polynomial Jastrow

The cutoff-polynomial Jastrow family with linear cusp constraints
({ref}`sec:Jastrow_DTN`):

- [N. D. Drummond, M. D. Towler, and R. J. Needs, Phys. Rev. B 70, 235119 (2004)](https://doi.org/10.1103/PhysRevB.70.235119)

(reference_agp)=

### Antisymmetrized Geminal Power (AGP)

The AGP and its Jastrow-correlated variants (JAGP, AGPn; {ref}`sec:AGP`):

- [M. Casula and S. Sorella, J. Chem. Phys. 119, 6500 (2003)](https://doi.org/10.1063/1.1604379)
- [M. Casula, C. Attaccalite, and S. Sorella, J. Chem. Phys. 121, 7110 (2004)](https://doi.org/10.1063/1.1794632)
- [M. Marchi, S. Azadi, M. Casula, and S. Sorella, J. Chem. Phys. 131, 154116 (2009)](https://doi.org/10.1063/1.3249966)

(reference_periodic_orbitals)=

### Periodic one-particle representations

B-spline (blip) representation of Bloch orbitals ({ref}`sec:atomic_orbitals`):

- [D. Alfe and M. J. Gillan, Phys. Rev. B 70, 161101(R) (2004)](https://doi.org/10.1103/PhysRevB.70.161101)

Twist averaging of periodic observables:

- [C. Lin, F. H. Zong, and D. M. Ceperley, Phys. Rev. E 64, 016702 (2001)](https://doi.org/10.1103/PhysRevE.64.016702)

## Hamiltonian

(reference_ecp)=

### Effective core potentials

Angular quadrature for the non-local pseudopotential
({ref}`sec:Coulomb-and-ECP`):

- [L. Mitas, E. L. Shirley, and D. M. Ceperley, J. Chem. Phys. 95, 3467 (1991)](https://doi.org/10.1063/1.460849)

(reference_mpc)=

### Model periodic Coulomb interaction (MPC)

The finite-size correction of the electron-electron interaction:

- [L. M. Fraser, W. M. C. Foulkes, G. Rajagopal, R. J. Needs, S. D. Kenny, and A. J. Williamson, Phys. Rev. B 53, 1814 (1996)](https://doi.org/10.1103/PhysRevB.53.1814)
- [A. J. Williamson, G. Rajagopal, R. J. Needs, L. M. Fraser, W. M. C. Foulkes, Y. Wang, and M.-Y. Chou, Phys. Rev. B 55, R4851 (1997)](https://doi.org/10.1103/PhysRevB.55.R4851)
- [P. R. C. Kent, R. Q. Hood, A. J. Williamson, R. J. Needs, W. M. C. Foulkes, and G. Rajagopal, Phys. Rev. B 59, 1917 (1999)](https://doi.org/10.1103/PhysRevB.59.1917)

## Variational Monte Carlo

(reference_metropolis)=

### Sampling and regularization

The Attaccalite--Sorella reweighting used to regularize the sampling near the
nodal surface ({ref}`vmc_tags`):

- [C. Attaccalite and S. Sorella, Phys. Rev. Lett. 100, 114501 (2008)](https://doi.org/10.1103/PhysRevLett.100.114501)

(reference_stochastic_reconfiguration)=

### Stochastic Reconfiguration

Wave functions are optimized with the Stochastic Reconfiguration (natural
gradient) method ({doc}`notes/wavefunction_optimization`):

- [S. Sorella, Phys. Rev. Lett. 80, 4558 (1998)](https://doi.org/10.1103/PhysRevLett.80.4558)
- [S. Sorella, Phys. Rev. B 64, 024512 (2001)](https://doi.org/10.1103/PhysRevB.64.024512)
- [S. Sorella, Phys. Rev. B 71, 241103(R) (2005)](https://doi.org/10.1103/PhysRevB.71.241103)
- [S. Sorella, M. Casula, and D. Rocca, J. Chem. Phys. 127, 014105 (2007)](https://doi.org/10.1063/1.2746035)

(reference_linear_method)=

### Linear method and accelerated SR

The linear method used as an alternative to plain SR, and the linear-method
inspired adaptive step:

- [M. P. Nightingale and V. Melik-Alaverdian, Phys. Rev. Lett. 87, 043401 (2001)](https://doi.org/10.1103/PhysRevLett.87.043401)
- [C. J. Umrigar, J. Toulouse, C. Filippi, S. Sorella, and R. G. Hennig, Phys. Rev. Lett. 98, 110201 (2007)](https://doi.org/10.1103/PhysRevLett.98.110201)
- [J. Toulouse and C. J. Umrigar, J. Chem. Phys. 126, 084102 (2007)](https://doi.org/10.1063/1.2437215)
- [S. Sorella, M. Casula, and D. Rocca, J. Chem. Phys. 127, 014105 (2007)](https://doi.org/10.1063/1.2746035)

## Diffusion Monte Carlo

(reference_lrdmc)=

### Diffusion Monte Carlo Method with Lattice Regularization (LRDMC)

Two LRDMC algorithms are implemented in jQMC ({doc}`notes/lrdmc`). The first
is the original LRDMC algorithm, exposed as `lrdmc-tau`:

- [M. Casula, C. Filippi, and S. Sorella, Phys. Rev. Lett. 95, 100201 (2005)](https://doi.org/10.1103/PhysRevLett.95.100201)
- [M. Casula, Phys. Rev. B 74, 161102(R) (2006)](https://doi.org/10.1103/PhysRevB.74.161102)
- [M. Casula, S. Moroni, S. Sorella, and C. Filippi, J. Chem. Phys. 132, 154113 (2010)](https://doi.org/10.1063/1.3380831)
- [K. Nakano, R. Maezono, and S. Sorella, Phys. Rev. B 101, 155106 (2020)](https://doi.org/10.1103/PhysRevB.101.155106)

The second, exposed as `lrdmc-bra`, is a modified formulation designed to
maintain better load balancing across walkers:

- [K. Nakano, S. Sorella, and M. Casula, J. Chem. Phys. 163, 194117 (2025)](https://doi.org/10.1063/5.0296986)

Although the latter algorithm (`lrdmc-bra`) offers superior parallel
efficiency, the two are different projection schemes (a fixed number of
projections versus a fixed imaginary time per measurement) that converge to
the same fixed-node energy, and they have been verified to yield consistent
results.

The Green's function Monte Carlo background of both algorithms:

- [D. F. B. ten Haaf, H. J. M. van Bemmel, J. M. J. van Leeuwen, W. van Saarloos, and D. M. Ceperley, Phys. Rev. B 51, 13039 (1995)](https://doi.org/10.1103/PhysRevB.51.13039)
- [M. Calandra Buonaura and S. Sorella, Phys. Rev. B 57, 11446 (1998)](https://doi.org/10.1103/PhysRevB.57.11446)
- [S. Sorella and L. Capriotti, Phys. Rev. B 61, 2599 (2000)](https://doi.org/10.1103/PhysRevB.61.2599)
- [N. Trivedi and D. M. Ceperley, Phys. Rev. B 41, 4552 (1990)](https://doi.org/10.1103/PhysRevB.41.4552)

Textbook treatment of the algorithms and of the error analysis
({doc}`notes/error_analysis`):

- [F. Becca and S. Sorella, *Quantum Monte Carlo Approaches for Correlated Systems*, Cambridge University Press (2017)](https://doi.org/10.1017/9781316417041)

## Energy derivatives

(reference_forces)=

### Atomic forces and stress in VMC

Adjoint algorithmic differentiation of the local energy and the space-warp
coordinate transformation ({ref}`subsec:forces_mcmc`):

- [C. J. Umrigar, Int. J. Quantum Chem. Symp. 23, 217 (1989)](https://doi.org/10.1002/qua.560360826)
- [C. Filippi and C. J. Umrigar, Phys. Rev. B 61, R16291 (2000)](https://doi.org/10.1103/PhysRevB.61.R16291)
- [S. Sorella and L. Capriotti, J. Chem. Phys. 133, 234111 (2010)](https://doi.org/10.1063/1.3516208)
- [K. Nakano, A. Raghav, and S. Sorella, J. Chem. Phys. 156, 034101 (2022)](https://doi.org/10.1063/5.0076302)
- [A. Griewank and A. Walther, *Evaluating Derivatives*, SIAM (2008)](https://doi.org/10.1137/1.9780898717761)

(reference_forces_lrdmc)=

### Atomic forces and stress in LRDMC

The mixed-distribution estimator and the regularization of the local-energy
derivatives near the nodal surface ({ref}`subsec:forces_lrdmc`):

- [P. J. Reynolds, R. N. Barnett, B. L. Hammond, R. M. Grimes, and W. A. Lester, Int. J. Quantum Chem. 29, 589 (1986)](https://doi.org/10.1002/qua.560290403)
- [S. Pathak and L. K. Wagner, AIP Advances 10, 085213 (2020)](https://doi.org/10.1063/5.0004008)
- [K. Nakano, S. Sorella, D. Alfe, and A. Zen, J. Chem. Theory Comput. 20, 4591 (2024)](https://doi.org/10.1021/acs.jctc.4c00139)

## Software

(reference_jax)=

### JAX and TREXIO

- [J. Bradbury et al., JAX: composable transformations of Python+NumPy programs (2018)](https://github.com/google/jax)
- [E. Posenitskiy et al., J. Chem. Phys. 158, 174801 (2023)](https://doi.org/10.1063/5.0148161) (TREXIO)

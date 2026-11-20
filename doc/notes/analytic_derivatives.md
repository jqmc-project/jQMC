# Analytic derivative backend

jQMC uses analytic (closed-form) derivatives for production MCMC and
GFMC/LRDMC observable derivatives: parameter derivatives for the
optimization, forces, and stress. On CPU, calculations use the Rust kernels
by default; XLA is used on accelerators and when CPU Rust execution is
disabled with `JQMC_FLAG_RUST_ON_CPU=0`. Both backends run the same analytic
kernels with the same precision rules, so switching backend does not change
the estimator.

The analytic route never silently switches an individual derivative to
automatic differentiation. If a requested derivative has no analytic
implementation (for example, basis-exponent optimization for an AO type
without analytic radial derivatives), jQMC raises a `NotImplementedError`
instead. This matters when comparing performance as well as numerical
results: a run that silently changed derivative implementation would not be
a valid backend measurement.

## Validation

In full precision, the analytic derivatives are checked against a separately
implemented automatic-differentiation reference; in mixed precision, against
tolerances matched to the reduced-precision parts of the calculation. The
comparison is made at fixed configurations first, because native and XLA
random streams need not produce identical trajectories. The test routes and
error criteria are not described here; see {doc}`mixed_precision` for the zone
policy and the tolerance helpers.

## Interpreting performance

Compare backends using the same checkout, input, device, precision mode,
walker count, and number of steps. Report compilation and steady-state step
time separately. Small CPU calculations can be dominated by fixed overhead,
whereas accelerator measurements need enough post-warmup steps to amortize
compilation and data transfer.

Setting `JQMC_FLAG_RUST_ON_CPU=0` selects the XLA execution path on CPU; it
does not change the scientific estimator. See {doc}`../execution` for the
backend selection table.

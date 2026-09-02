"""
Factory function for creating the scalar-transport (thermal) field arrays.

Returns the scalar distribution-function pair (*g_0*, *g_1*), the boundary-condition mask,
and the missing-population mask, all allocated on the given grid and backend.

Also provides the mapping between the lattice relaxation parameter and the scalar
diffusivity, which is the thermal counterpart of ``omega = 1 / (3 * nu + 0.5)``.
"""

from xlb import DefaultConfig
from xlb.grid import grid_factory
from xlb.precision_policy import Precision
from xlb.compute_backend import ComputeBackend
from typing import Tuple


def create_thermal_fields(
    grid_shape: Tuple[int, int, int] = None,
    grid=None,
    velocity_set=None,
    compute_backend=None,
    precision_policy=None,
):
    """Create fields for a scalar-transport (advection-diffusion) solver.

    Args:
        grid_shape: Tuple of grid dimensions. Required if grid is not provided.
        grid: Optional Grid object. If provided, will be used instead of creating new grid.
        velocity_set: Optional velocity set for the scalar populations. This is usually a
                      reduced stencil (e.g. D2Q5) and need not match the stencil used by
                      the flow solver. Defaults to DefaultConfig.velocity_set.
        compute_backend: Optional compute backend. Defaults to DefaultConfig.default_backend.
        precision_policy: Optional precision policy. Defaults to DefaultConfig.default_precision_policy.

    Returns:
        Tuple of (grid, g_0, g_1, missing_mask, bc_mask)
    """
    velocity_set = velocity_set or DefaultConfig.velocity_set
    compute_backend = compute_backend or DefaultConfig.default_backend
    precision_policy = precision_policy or DefaultConfig.default_precision_policy

    if grid is None:
        if grid_shape is None:
            raise ValueError("grid_shape must be provided when grid is None")
        grid = grid_factory(grid_shape, compute_backend=compute_backend, velocity_set=velocity_set)

    # Create fields
    g_0 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    g_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    bc_mask = grid.create_field(cardinality=1, dtype=Precision.UINT8)
    if compute_backend in [ComputeBackend.WARP, ComputeBackend.NEON]:
        # For WARP and NEON, we use UINT8 for missing mask
        missing_mask = grid.create_field(cardinality=velocity_set.q, dtype=Precision.UINT8)
    else:
        # For JAX, we use bool for missing mask
        missing_mask = grid.create_field(cardinality=velocity_set.q, dtype=Precision.BOOL)

    return grid, g_0, g_1, missing_mask, bc_mask


def omega_from_diffusivity(diffusivity, cs2=1.0 / 3.0):
    """Relaxation parameter of the scalar populations for a given lattice diffusivity.

    Chapman-Enskog analysis of the BGK advection-diffusion equation gives
    ``diffusivity = cs^2 * (1 / omega - 1/2)``, hence::

        omega = 1 / (diffusivity / cs^2 + 1/2)

    Args:
        diffusivity: Scalar diffusivity in lattice units. For the heat equation this is
                     ``alpha * dt / dx^2`` with ``alpha = k / (rho * cp)``.
        cs2: Lattice speed of sound squared. Defaults to 1/3, the value used by all XLB
             velocity sets.

    Returns:
        The relaxation parameter omega, which must lie in (0, 2) for stability.
    """
    return 1.0 / (diffusivity / cs2 + 0.5)


def diffusivity_from_omega(omega, cs2=1.0 / 3.0):
    """Lattice diffusivity of the scalar populations for a given relaxation parameter.

    Inverse of :func:`omega_from_diffusivity`.

    Args:
        omega: Relaxation parameter of the scalar populations.
        cs2: Lattice speed of sound squared. Defaults to 1/3.

    Returns:
        The scalar diffusivity in lattice units.
    """
    return cs2 * (1.0 / omega - 0.5)

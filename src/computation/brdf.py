"""
BRDF (Bidirectional Reflectance Distribution Function) calculations and material management.

Provides:
- Ashikhmin-Shirley BRDF model calculations
- Material parameter management for satellite components
- Flux-to-magnitude conversion utilities
"""

import numpy as np
import logging
from typing import Dict, Optional
from dataclasses import dataclass

from ..io.stl_loader import Satellite, Facet, BRDFMaterialProperties
from ..config.rso_config_schemas import RSO_Config, BRDFParameters

logger = logging.getLogger(__name__)

# Physical constants
SUN_APPARENT_MAGNITUDE = -26.74  # m0 from reference


@dataclass
class LightCurveResult:
    """
    Container for light curve generation results.

    Attributes:
        epochs: Array of epoch times (ET)
        magnitudes: Array of apparent magnitudes
        total_flux: Array of total flux values
        observer_distances: Array of observer distances (meters)
    """
    epochs: np.ndarray
    magnitudes: np.ndarray
    total_flux: np.ndarray
    observer_distances: np.ndarray


def convert_flux_to_magnitude(flux_value: float, observer_distance_km: float,
                              mode: str = 'normal') -> float:
    """
    Convert flux or log-flux to apparent magnitude.

    Args:
        flux_value: Either sum of flux numerators (normal mode) or
                    log10(sum of flux numerators) (surrogate mode)
        observer_distance_km: Distance from satellite to observer in kilometers
        mode: 'normal' for linear flux sum, 'surrogate' for log10 flux sum

    Returns:
        Apparent magnitude

    Raises:
        ValueError: If mode is invalid or inputs are invalid
    """
    if observer_distance_km <= 0:
        raise ValueError(f"Observer distance must be positive: {observer_distance_km} km")

    # Convert km to meters for flux calculation
    distance_m = observer_distance_km * 1000.0
    five_log_d = 5.0 * np.log10(distance_m)

    if mode == 'normal':
        if flux_value > 1e-20:
            magnitude = SUN_APPARENT_MAGNITUDE + five_log_d - 2.5 * np.log10(flux_value)
        else:
            magnitude = np.inf
    elif mode == 'surrogate':
        if not np.isfinite(flux_value):
            raise ValueError(f"Invalid logged flux value: {flux_value}")
        magnitude = SUN_APPARENT_MAGNITUDE + five_log_d - 2.5 * flux_value
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'normal' or 'surrogate'")

    return magnitude


class BRDFCalculator:
    """
    BRDF calculator implementing the Ashikhmin-Shirley model.

    Handles pure BRDF calculations for satellite facets.
    Does NOT handle articulation or shadow computation - those are done separately.
    """

    def __init__(self):
        """Initialize the BRDF calculator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def calculate_brdf(self,
                      n_dot_k1: float,
                      n_dot_k2: float,
                      h_dot_k1: float,
                      n_dot_h: float,
                      material: BRDFMaterialProperties) -> float:
        """
        Calculate the Ashikhmin-Shirley BRDF value.

        Args:
            n_dot_k1: Dot product of surface normal and light direction
            n_dot_k2: Dot product of surface normal and observer direction
            h_dot_k1: Dot product of halfway vector and light direction
            n_dot_h: Dot product of surface normal and halfway vector
            material: BRDF material properties

        Returns:
            Total BRDF value (rho)
        """
        # Calculate intermediate terms for diffuse component
        alpha = 1.0 - n_dot_k1 / 2.0
        beta = 1.0 - n_dot_k2 / 2.0

        # Calculate Fresnel term for specular component
        fresnel = material.r_s + (1.0 - material.r_s) * (1.0 - h_dot_k1)**5

        # Diffuse component
        rho_diff = (28.0 * material.r_d / (23.0 * np.pi)) * (1.0 - material.r_s) * \
                   (1.0 - alpha**5) * (1.0 - beta**5)

        # Specular component
        denominator = h_dot_k1 * max(n_dot_k1, n_dot_k2)
        if denominator > 1e-10:
            rho_spec = ((material.n_phong + 1.0) / (8.0 * np.pi)) * \
                       (n_dot_h**material.n_phong / denominator) * fresnel
        else:
            rho_spec = 0.0

        return rho_diff + rho_spec

    def calculate_facet_flux_numerator(self,
                                       facet: Facet,
                                       sun_direction: np.ndarray,
                                       observer_direction: np.ndarray) -> float:
        """
        Calculate the flux numerator for a single facet (without distance scaling).

        Returns rho * A * (n.k1) * (n.k2) without dividing by distance^2.
        Used for efficient batch processing where distance is applied once at the end.

        Note: This should only be called for lit facets. Shadow status is handled
        by the caller (lightcurve_generator) which filters facets before calling.

        Args:
            facet: Facet object with geometry and material properties
            sun_direction: Unit vector from facet to sun (k1)
            observer_direction: Unit vector from facet to observer (k2)

        Returns:
            Flux numerator: rho * A * (n.k1) * (n.k2), or 0.0 if facet is not visible
        """
        # Calculate dot products
        n_dot_k1 = np.dot(facet.normal, sun_direction)
        n_dot_k2 = np.dot(facet.normal, observer_direction)

        # Visibility check: facet must face both sun and observer
        if n_dot_k1 <= 0 or n_dot_k2 <= 0:
            return 0.0

        # Calculate halfway vector
        halfway = sun_direction + observer_direction
        halfway_norm = np.linalg.norm(halfway)
        if halfway_norm > 1e-10:
            halfway = halfway / halfway_norm
        else:
            return 0.0

        h_dot_k1 = np.dot(halfway, sun_direction)
        n_dot_h = np.dot(facet.normal, halfway)

        # Calculate BRDF using Ashikhmin-Shirley model
        rho = self.calculate_brdf(n_dot_k1, n_dot_k2, h_dot_k1, n_dot_h,
                                  facet.material_properties)

        # Return flux numerator (without distance scaling)
        return rho * facet.area * n_dot_k1 * n_dot_k2

    def update_satellite_brdf_with_manager(self, satellite: Satellite, brdf_manager: 'BRDFManager'):
        """
        Update BRDF parameters using a BRDFManager.

        Args:
            satellite: Satellite model to update
            brdf_manager: BRDFManager instance with parameter mappings
        """
        brdf_manager.update_satellite_brdf_parameters(satellite)


class BRDFManager:
    """Manages BRDF material parameter assignment for satellite components."""

    def __init__(self, config: RSO_Config):
        """
        Initialize BRDF manager with configuration.

        Args:
            config: RSO configuration containing BRDF mappings
        """
        self.config = config
        self.brdf_mappings = config.brdf_mappings

    def get_brdf_parameters(self, component) -> BRDFMaterialProperties:
        """
        Get BRDF parameters for a component.

        Looks up BRDF parameters from the configuration's brdf_mappings.
        Supports exact name match and substring matching.

        Args:
            component: Component to get BRDF parameters for

        Returns:
            BRDFMaterialProperties for the component

        Raises:
            ValueError: If no BRDF parameters found for component
        """
        # First, try exact component name match
        if component.name in self.brdf_mappings:
            params = self.brdf_mappings[component.name]
            return BRDFMaterialProperties(
                r_d=params.r_d,
                r_s=params.r_s,
                n_phong=params.n_phong
            )

        # Second, try substring matching
        for key, params in self.brdf_mappings.items():
            if key in component.name:
                return BRDFMaterialProperties(
                    r_d=params.r_d,
                    r_s=params.r_s,
                    n_phong=params.n_phong
                )

        raise ValueError(
            f"No BRDF parameters found for component '{component.name}'. "
            f"Please add BRDF mapping in configuration file."
        )

    def update_satellite_brdf_parameters(self, satellite: Satellite) -> None:
        """
        Update BRDF parameters for all components in a satellite.

        Args:
            satellite: Satellite model to update
        """
        for component in satellite.components:
            brdf_params = self.get_brdf_parameters(component)

            # Update component default material
            if component.default_material is None:
                component.default_material = brdf_params
            else:
                component.default_material.r_d = brdf_params.r_d
                component.default_material.r_s = brdf_params.r_s
                component.default_material.n_phong = brdf_params.n_phong

            # Update all facets in this component
            for facet in component.facets:
                facet.material_properties.r_d = brdf_params.r_d
                facet.material_properties.r_s = brdf_params.r_s
                facet.material_properties.n_phong = brdf_params.n_phong

    def validate_brdf_parameters(self) -> bool:
        """
        Validate BRDF parameters for physical correctness.

        Returns:
            True if all parameters are valid, False otherwise
        """
        for component_name, params in self.brdf_mappings.items():
            if not (0.0 <= params.r_d <= 1.0):
                return False
            if not (0.0 <= params.r_s <= 1.0):
                return False
            if params.n_phong < 0.0:
                return False
            if params.r_d + params.r_s > 1.0:
                return False
        return True


# =============================================================================
# Vectorized BRDF Functions (for ~50-200x speedup)
# =============================================================================

def calculate_brdf_vectorized(
    n_dot_k1: np.ndarray,
    n_dot_k2: np.ndarray,
    h_dot_k1: np.ndarray,
    n_dot_h: np.ndarray,
    r_d: np.ndarray,
    r_s: np.ndarray,
    n_phong: np.ndarray
) -> np.ndarray:
    """
    Vectorized Ashikhmin-Shirley BRDF calculation.

    Computes BRDF for multiple facets simultaneously using NumPy broadcasting.
    Provides ~50-200x speedup over the scalar implementation.

    Args:
        n_dot_k1: Normal dot sun direction (M,)
        n_dot_k2: Normal dot observer direction (M,)
        h_dot_k1: Halfway dot sun direction (M,) or scalar
        n_dot_h: Normal dot halfway vector (M,)
        r_d: Diffuse reflectivity per facet (M,)
        r_s: Specular reflectivity per facet (M,)
        n_phong: Phong exponent per facet (M,)

    Returns:
        BRDF values (M,)
    """
    # Diffuse component (Ashikhmin-Shirley)
    alpha = 1.0 - n_dot_k1 / 2.0
    beta = 1.0 - n_dot_k2 / 2.0

    rho_diff = (28.0 * r_d / (23.0 * np.pi)) * (1.0 - r_s) * \
               (1.0 - alpha**5) * (1.0 - beta**5)

    # Specular component with Fresnel term
    fresnel = r_s + (1.0 - r_s) * (1.0 - h_dot_k1)**5

    # Denominator with max(n_dot_k1, n_dot_k2)
    max_n_dot = np.maximum(n_dot_k1, n_dot_k2)
    denominator = h_dot_k1 * max_n_dot

    # Avoid division by zero using np.where
    safe_denom = np.where(denominator > 1e-10, denominator, 1.0)

    # Specular term: ((n+1)/(8*pi)) * (n_dot_h^n / denom) * fresnel
    rho_spec = np.where(
        denominator > 1e-10,
        ((n_phong + 1.0) / (8.0 * np.pi)) * (n_dot_h**n_phong / safe_denom) * fresnel,
        0.0
    )

    return rho_diff + rho_spec


def calculate_flux_vectorized(
    normals: np.ndarray,
    areas: np.ndarray,
    r_d: np.ndarray,
    r_s: np.ndarray,
    n_phong: np.ndarray,
    sun_direction: np.ndarray,
    observer_direction: np.ndarray,
    lit_mask: np.ndarray
) -> float:
    """
    Calculate total flux for all facets using vectorized operations.

    This function eliminates the inner facet loop in light curve generation,
    providing ~50-200x speedup for BRDF calculations.

    Args:
        normals: Facet normals (M, 3) - already transformed for articulation
        areas: Facet areas (M,)
        r_d: Diffuse reflectivity (M,)
        r_s: Specular reflectivity (M,)
        n_phong: Phong exponent (M,)
        sun_direction: Sun direction vector (3,)
        observer_direction: Observer direction vector (3,)
        lit_mask: Boolean mask for lit facets (M,) - True = not shadowed

    Returns:
        Total flux (sum over all visible, lit facets)
    """
    M = len(normals)

    if M == 0:
        return 0.0

    # Vectorized dot products: normals (M, 3) @ direction (3,) -> (M,)
    n_dot_k1 = normals @ sun_direction      # (M,)
    n_dot_k2 = normals @ observer_direction  # (M,)

    # Visibility mask: facet must face both sun and observer
    visible = (n_dot_k1 > 0) & (n_dot_k2 > 0)

    # Combined mask: visible AND lit (not shadowed)
    active = visible & lit_mask

    # Early exit if nothing is active
    if not np.any(active):
        return 0.0

    # Halfway vector (same for all facets at this epoch)
    halfway = sun_direction + observer_direction
    halfway_norm = np.linalg.norm(halfway)
    if halfway_norm < 1e-10:
        return 0.0
    halfway = halfway / halfway_norm

    # Dot products with halfway vector
    h_dot_k1 = np.dot(halfway, sun_direction)  # Scalar - same for all facets
    n_dot_h = normals @ halfway  # (M,)

    # Vectorized BRDF calculation
    rho = calculate_brdf_vectorized(
        n_dot_k1, n_dot_k2, h_dot_k1, n_dot_h,
        r_d, r_s, n_phong
    )

    # Flux contribution per facet: rho * area * n_dot_k1 * n_dot_k2
    flux = rho * areas * n_dot_k1 * n_dot_k2

    # Sum only active facets (visible AND lit)
    return np.sum(flux[active])

# Core computational functions for satellite light curve generation

from .brdf import (
    BRDFCalculator,
    BRDFManager,
    convert_flux_to_magnitude,
    LightCurveResult,
    calculate_brdf_vectorized,
    calculate_flux_vectorized,
)

from .lightcurve_generator import generate_lightcurves

from .facet_data_extractor import (
    FacetArrays,
    extract_facet_arrays,
    apply_articulation_to_arrays,
    apply_articulation_to_vertices,
    lit_status_to_flat_array,
    flat_array_to_lit_status_dict,
)

__all__ = [
    # BRDF
    'BRDFCalculator',
    'BRDFManager',
    'convert_flux_to_magnitude',
    'LightCurveResult',
    'calculate_brdf_vectorized',
    'calculate_flux_vectorized',
    # Light curve generation
    'generate_lightcurves',
    # Facet data extraction
    'FacetArrays',
    'extract_facet_arrays',
    'apply_articulation_to_arrays',
    'apply_articulation_to_vertices',
    'lit_status_to_flat_array',
    'flat_array_to_lit_status_dict',
]
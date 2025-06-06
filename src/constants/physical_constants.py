from astropy.constants import R_earth, M_earth

PHYSICAL_CONSTANTS = {
    'earth_radius': R_earth.value,
    'earth_mass': M_earth.value,
    'hz_flux_range': (0.36, 1.1),
    'raidus_valley': 1.6,  # Fulton et al. 2017
    'teff_solar': 5578,
    'temp_habitable': (273, 373)
}
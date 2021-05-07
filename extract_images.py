import cpyMSpec
import numpy as np
import pandas as pd
from pyMSpec.pyisocalc.pyisocalc import parseSumFormula
from pyimzml.ImzMLParser import ImzMLParser

assert cpyMSpec.utils.VERSION <= '0.4.2', 'Using an old version of cpyMSpec.'

ANALYZERS = ('tof', 'orbitrap', 'ft-icr')
TOL_MODES = ('constant_fwhm', 'tof', 'orbitrap', 'ft-icr')
DEFAULT_ANALYZER = 'orbitrap'
DEFAULT_TOL_MODE = 'tof'  # With TOF, 1ppm is always mz/1000000
DEFAULT_TOL_PPM = 3
DEFAULT_RP = 140000
DEFAULT_BASE_MZ = 200.0
DEFAULT_ADDUCTS = ('+H', '+K', '+Na', '')
DEFAULT_CHARGE = 1


def ppm_to_daltons(mzs, ppm=DEFAULT_TOL_PPM, tol_mode=DEFAULT_TOL_MODE, base_mz=DEFAULT_BASE_MZ):
    """
    Calculates how big "N ppm" is in Daltons for the specified scaling mode. e.g. with base_mz=100:
    All modes: 1ppm @ 100 => 0.1 mDa (1ppm)
    for constant FWHM: 1ppm @ 800 => 0.1 mDa (0.25ppm)
    for TOF: 1ppm @ 800 => 0.8 mDa (1ppm)
    for Orbitrap: 1ppm @ 800 => 1.6 mDa (2ppm)
    for FT-ICR: 1ppm @ 800 => 3.2 mDa (4ppm)
    """
    assert tol_mode in TOL_MODES, f'invalid tol_mode: {tol_mode}'
    base_tol = base_mz * ppm * 1e-6

    if tol_mode == 'constant_fwhm':
        return base_tol if np.isscalar(mzs) else np.full_like(mzs, base_tol)
    elif tol_mode == 'tof':
        return base_tol * (mzs / base_mz)
    elif tol_mode == 'orbitrap':
        return base_tol * (mzs / base_mz) ** 1.5
    else: # tol_mode == 'ft-icr':
        return base_tol * (mzs / base_mz) ** 2


def tol_edges(mzs, tol_ppm=DEFAULT_TOL_PPM, tolerance_scaling=DEFAULT_TOL_MODE, base_mz=DEFAULT_BASE_MZ):
    """Calculates the upper and lower m/z values for tolerance windows around mzs,
    using the supplied ppm & scaling mode"""
    half_width = ppm_to_daltons(mzs, tol_ppm, tolerance_scaling, base_mz)
    return mzs - half_width, mzs + half_width


def safeParseSumFormula(formula):
    """Wrapper for parseSumFormula to catch exceptions"""
    try:
        return str(parseSumFormula(formula))
    except Exception:
        return None


def load_db(db_path, default_adducts=DEFAULT_ADDUCTS, default_charge=DEFAULT_CHARGE):
    """Loads a database CSV or TSV file, applys default adducts and a charge if no adducts/charge
     are specified in the file"""
    if '\t' in open(db_path).readline():
        db = pd.read_csv(db_path, sep='\t')
    else:
        db = pd.read_csv(db_path)

    assert 'formula' in db.columns, 'Database file missing "formula" column'

    if 'adduct' not in db.columns:
        db = pd.concat([db.assign(adduct=adduct) for adduct in default_adducts], ignore_index=True)
    if 'charge' not in db.columns:
        db = db.assign(charge=default_charge)

    db['ion_formula'] = [
        safeParseSumFormula(formula + adduct)
        for formula, adduct in db[['formula', 'adduct']].itertuples(False, None)
    ]

    return db


def calculate_centroids(db, n_peaks=1, min_abundance=0, analyzer=DEFAULT_ANALYZER, rp=DEFAULT_RP, base_mz=DEFAULT_BASE_MZ):
    """Calculates the isotopic peak centroids for items in the supplied database.
    Database rows should have 'ion_formula' and 'charge' columns.
    If n_peaks > 1, multiple rows per ion_formula/charge will be returned.
    If an ion formula creates an invalid molecule (e.g. H2O-Cl), it will be

    """
    assert analyzer in ANALYZERS
    assert 'ion_formula' in db.columns, 'db must have ion_formula and charge columns'
    assert 'charge' in db.columns, 'db must have ion_formula and charge columns'

    # Remove the hyphen from ft-icr if needed, because cpyMSpec only recognizes "fticr"
    instrument_model = cpyMSpec.InstrumentModel(analyzer.replace('-', ''), rp, base_mz)
    results = []

    for ion_formula, charge in db[['ion_formula', 'charge']].drop_duplicates().itertuples(False, None):
        if not ion_formula or not charge:
            continue
        try:
            iso_pattern = cpyMSpec.isotopePattern(ion_formula)
            iso_pattern.addCharge(charge)
            centroids = iso_pattern.centroids(instrument_model, min_abundance=min_abundance)
            for peak_i, mass, ints in zip(range(n_peaks), centroids.masses, centroids.intensities):
                # Divide mass by charge, because centroids has masses, not mass-to-charge-ratios
                results.append((ion_formula, charge, peak_i, mass / abs(charge), ints))
        except Exception as ex:
            if 'total number of' in str(ex) and 'less than zero' in str(ex):
                # Invalid molecule due to adduct removing non-existent elements
                continue
            raise Exception(f'Could not generate isotopic pattern for {ion_formula} (charge: {charge})')

    return pd.DataFrame(results, columns=['ion_formula', 'charge', 'peak_i', 'mz', 'db_ints'])


def extract_peaks(
    imzml_path,
    db,
    tol_ppm=DEFAULT_TOL_PPM,
    tol_mode=DEFAULT_TOL_MODE,
    base_mz=DEFAULT_BASE_MZ,
):
    """
    Extract all peaks from the given imzML file for the supplied database of molecules.
    :param imzml_path:
    :param db: A pandas DataFrame containing an 'mz' column. Additional metadata columns are also allowed.
    :param tol_ppm:
        The maximum distance from a theoretical m/z to search for peaks. e.g. 3 means +/- 3ppm
    :param tol_mode:
        The model for adjusting tol_ppm based on the area of the mass range.
        To match METASPACE, specify 'tof', which means 1ppm is always mz * 1e-6 (i.e. 1ppm at every mass)
        See the `ppm_to_daltons` function for more examples.
    :param base_mz:
        The base m/z for tolerance calculations. Doesn't matter with 'tof'.
        See the `ppm_to_daltons` function for more details.
    :return:
        coords_df - a DataFrame mapping spectrum idx to x,y values.
            Needed for converting 'peaks_df' values to images
        peaks - A list of dicts. Each dict contains:
            'mol': A NamedTuple of the DB peak row. Access fields with e.g. peak['mol'].formula
            'peaks_df': a DataFrame with one row per found peak. Columns:
                'sp': Spectrum idx
                'mz': m/z
                'ints': Intensity value
    """
    assert 'mz' in db.columns, 'db must have an "mz" column'
    assert tol_mode in TOL_MODES, f'invalid tol_mode: {tol_mode}'
    p = ImzMLParser(str(imzml_path))
    coords_df = pd.DataFrame(p.coordinates, columns=['x', 'y', 'z'][:len(p.coordinates[0])], dtype='i')
    coords_df['x'] -= np.min(coords_df.x)
    coords_df['y'] -= np.min(coords_df.y)

    mz_tol_lo, mz_tol_hi = tol_edges(db.mz, tol_ppm, tol_mode, base_mz)
    # Uncomment this to add the tolerance boundaries to db for debugging:
    # db['mz_tol_lo'], db['mz_tol_hi'] = mz_tol_lo, mz_tol_hi

    mol_peaks = [[] for sp in range(len(coords_df))]

    for sp, x, y in coords_df[['x', 'y']].itertuples(True, None):
        mzs, ints = p.getspectrum(sp)
        mz_range_lo = np.searchsorted(mzs, mz_tol_lo, 'left')
        mz_range_hi = np.searchsorted(mzs, mz_tol_hi, 'right')
        mask = mz_range_lo != mz_range_hi
        for peak, idx_lo, idx_hi in zip(np.flatnonzero(mask), mz_range_lo[mask], mz_range_hi[mask]):
            for i in range(idx_lo, idx_hi):
                mol_peaks[peak].append((sp, mzs[i], ints[i]))

    empty_peaks_df = pd.DataFrame({
        'sp': pd.Series(dtype='i'),
        'mz': pd.Series(dtype='f'),
        'ints': pd.Series(dtype='f'),
    })

    result = [{
        'mol': db_row,
        'peaks_df': pd.DataFrame(peaks, columns=['sp', 'mz', 'ints']) if peaks else empty_peaks_df
    } for db_row, peaks in zip(db.itertuples(), mol_peaks)]

    return coords_df, result


def peaks_df_to_images(coords_df, peaks_df):
    """
    Creates m/z and intensity images from a peaks_df DataFrame from extract_peaks.

    :return: A tuple of two images:
    mz_image: a 2D numpy array with m/z values. Missing pixels are NaNs.
        If multiple peaks were in the tolerance window, they are averaged.
    ints_image: a 2D numpy array with intensity values. Missing pixels are 0.
        If multiple peaks were in the tolerance window, they are summed.
    """
    w, h = np.max(coords_df.x) + 1, np.max(coords_df.y) + 1
    mz_image = np.full((w, h), np.nan, dtype='f')
    ints_image = np.zeros((w, h), dtype='f')

    merged_df = coords_df.merge(peaks_df, left_index=True, right_on='sp')

    mz_image[merged_df.x, merged_df.y] = merged_df.mz
    ints_image[merged_df.x, merged_df.y] = merged_df.ints

    # The above code doesn't handle cases where multiple peaks affect the same pixel correctly.
    # Re-do pixels that have more than 1 peak. This code is much slower, so it's only used on the
    # spectra that need it.
    duplicated_peaks = merged_df[merged_df.duplicated(['x', 'y'], keep=False)]
    for (x, y), grp in duplicated_peaks.groupby(['x','y']):
        if len(grp) > 1:
            mz_image[x, y] = np.average(grp.mz, weights=grp.ints)
            ints_image[x, y] = np.sum(grp.ints)
        else:
            mz_image[x, y] = grp.mz.iloc[0]
            ints_image[x, y] = grp.ints.iloc[0]

    return mz_image, ints_image


def search_imzml_for_database_peaks(
    imzml_path,
    db_path,
    tol_ppm=DEFAULT_TOL_PPM,
    tol_mode=DEFAULT_TOL_MODE,
    analyzer=DEFAULT_ANALYZER,
    rp=DEFAULT_RP,
    base_mz=DEFAULT_BASE_MZ,
    default_adducts=DEFAULT_ADDUCTS,
    default_charge=DEFAULT_CHARGE,
    n_peaks=1,
    min_abundance=0,
):
    """
    Searches an ImzML file for all peaks corresponding to the molecules in the database file.

    :param imzml_path: Path to the .imzML file
    :param db_path: Path to a CSV or TSV database with at minimum a "formula" column,
        but potentially also specifying "adduct" and "charge" if you want to customize this on a
        per-molecule basis. Alternatively, supply an 'mz' column to skip adducts/isotopic peaks calculation.
    :param tol_ppm:
        The maximum distance from a theoretical m/z to search for peaks. e.g. 3 means +/- 3ppm
    :param tol_mode:
        The model for adjusting tol_ppm based on the area of the mass range.
        To match METASPACE, specify 'tof', which means 1ppm is always mz * 1e-6 (i.e. 1ppm at every mass)
        See the `ppm_to_daltons` function for more examples.
    :param analyzer:
        The instrument type for calculating centroids for non-monoisotopic peaks (i.e. if n_peaks > 1).
        Valid options: 'tof', 'orbitrap', 'ft-icr'
    :param rp:
        Analyzer resolving power at `base_mz` for centroids for non-monoisotopic peaks.
        Only relevant if n_peaks > 1.
    :param base_mz:
        Base m/z value for tol_ppm scaling and rp scaling. Default: 200
    :param default_adducts:
        If there is no 'adduct' column in the DB, each formula will be tried with each adduct from
        this list. If there is an 'adduct' column, this will be ignored.
    :param default_charge:
        If there is no 'adduct' charge in the DB, each formula will use this charge.
        If there is a 'charge' column, this will be ignored.
    :param n_peaks:
        Maximum number of isotopic centroid peaks to search for. If this is 1, only the
        monoisotopic peak will be searched for.
    :param min_abundance:
        Minimum relative abundance of a secondary isotopic peak relative to the most abundant peak.
    :return: Returns 3 items:
        db - The DB loaded from CSV/TSV, with adducts/charge applied. If multiple adducts,
            or n_peaks > 1 is specified, there will be multiple rows per input formula.
        coords_df - a DataFrame mapping spectrum idx to x,y values.
            Needed for converting 'peaks_df' values to images
        peaks - A list of dicts. Each dict contains:
            'mol': A NamedTuple of the DB peak row. Access fields with e.g. peak['mol'].formula
            'peaks_df': a DataFrame with one row per found peak. Columns:
                'sp': Spectrum idx
                'mz': m/z
                'ints': Intensity value

    """
    assert tol_mode in TOL_MODES, f'invalid tol_mode: {tol_mode}'
    assert analyzer in ANALYZERS

    db = load_db(db_path, default_adducts, default_charge)
    if 'mz' not in db.columns:
        centroids = calculate_centroids(db, n_peaks, min_abundance, analyzer, rp, base_mz)
        # NOTE: This merge will exclude any rows from the original DB if no centroids were generated
        # e.g. due to an invalid adduct.
        db_with_centroids = db.merge(centroids, on=['ion_formula', 'charge'])
    else:
        db_with_centroids = db

    coords_df, peaks = extract_peaks(imzml_path, db_with_centroids, tol_ppm, tol_mode, base_mz)
    return db_with_centroids, coords_df, peaks
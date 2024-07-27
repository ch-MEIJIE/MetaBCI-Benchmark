from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.signal import sosfiltfilt
    from metabci.brainda.paradigms import SSVEP, P300, MotorImagery, aVEP
    from metabci.brainda.algorithms.decomposition import generate_filterbank


def channel_selection(key):
    if key == 'occipital_9':
        return ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']


def create_ssvep_paradigm(srate, channels, interval, events):
    paradigm = SSVEP(
        srate=srate,
        channels=channels,
        intervals=interval,
        events=events)

    def ssvep_datahook(X, y, meta, caches):
        filters = generate_filterbank(
            [[8, 90]], [[6, 95]], srate, order=4, rp=1
        )
        X = sosfiltfilt(filters[0], X, axis=-1)
        return X, y, meta, caches

    paradigm.register_data_hook(ssvep_datahook)

    return paradigm


def gen_ssvep_filterbank(srate, n_bands=3):
    wp = [[8*i, 90] for i in range(1, n_bands+1)]
    ws = [[8*i-2, 95] for i in range(1, n_bands+1)]
    filterbank = generate_filterbank(
        wp, ws, srate, order=4, rp=1)
    filterweights = np.arange(1, len(filterbank)+1)**(-1.25) + 0.25
    return filterbank, filterweights
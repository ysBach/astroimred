import pytest

pytest.importorskip("ccdproc")
pytest.importorskip("scipy")

import astroimred.imutil as imutil


def pytest_configure(config):
    # The legacy regression fixtures were generated against the non-numba path.
    imutil.set_use_numba(False)

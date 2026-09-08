API Reference
=============

.. autotable::
   :toctree: api

   astroimred._core
   astroimred._core.astropy_helpers
   astroimred._core.geometry
   astroimred._core.numeric
   astroimred._core.scales
   astroimred._core.system
   astroimred._core.types
   astroimred._core.wcs
   astroimred.external
   astroimred.external.sep
   astroimred.fitsmgmt
   astroimred.fitsmgmt.airmass
   astroimred.fitsmgmt.header
   astroimred.fitsmgmt.io
   astroimred.fitsmgmt.naming
   astroimred.fitsmgmt.table
   astroimred.fitsmgmt.wcs
   astroimred.imutil
   astroimred.imutil.ccdops
   astroimred.imutil.imstat
   astroimred.imutil.pixels
   astroimred.imutil.imarith
   astroimred.imutil.imcombine
   astroimred.imutil.imcopy
   astroimred.imutil.imsmooth
   astroimred.logging
   astroimred.reduction
   astroimred.reduction.cli
   astroimred.reduction.crrej
   astroimred.reduction.preproc

Use root-level imports such as ``air.load_ccd`` for short FITS/image scripts.
Canonical module paths live under
``astroimred._core``, ``astroimred.external``, ``astroimred.fitsmgmt``,
``astroimred.imutil``, and ``astroimred.viz``.

Optional visualization modules
------------------------------

The visualization helpers require ``astroimred[full]``.

.. autotable::
   :toctree: api

   astroimred.viz
   astroimred.viz.enhance
   astroimred.viz.imshow
   astroimred.viz.rvsf
   astroimred.viz.ticks

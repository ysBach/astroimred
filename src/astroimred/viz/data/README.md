# CometCIEF example image

Read `CometCIEF_test.fits.xz` with Python's `lzma` module and Astropy FITS.
The archive preserves the **entire original FITS byte stream**, including
all 801 × 801 float32 pixels, headers, and padding. No crop, quantization,
rescaling, or metadata correction was applied.

```python
import lzma
from importlib.resources import as_file, files

from astropy.io import fits

resource = files("astroimred.viz").joinpath("data/CometCIEF_test.fits.xz")
with as_file(resource) as path, lzma.open(path, "rb") as stream:
    image, header = fits.getdata(stream, header=True, memmap=False)
```

The CometCIEF tutorial gives the nucleus at `(400.5, 400.5)` in one-based
coordinates. Use **`center=(399.5, 399.5)`** with AIR's zero-based API.

## Provenance

The input is the user-supplied `cometcief_techniques/CometCIEF_test.fits`,
distributed as a sample by the [CometCIEF facility](https://cie.psi.edu/techniques.html).
Credit: Samarasinha, N. H., Martin, M. P., and Larson, S. M. (2013),
*Cometary Coma Image Enhancement Facility*.

**The passband is unresolved.** The tutorial identifies comet Hale-Bopp and
describes dust continuum, but the FITS header has `OBJECT='HBcn'` and a
`321cn.imh` processing history. The original header is preserved. This sample
does not establish whether inverse-radius compensation suits a particular passband.

The supplied files and download page do not specify a separate data
license. This provenance note does not assign AIR's software license to
the image.

## Compression check

Measured on the same original FITS file with standard lossless compressors:

| Format | Settings | Bytes |
| --- | --- | ---: |
| FITS | Original | 2,574,720 |
| gzip | `gzip -n -9` | 2,090,675 |
| bzip2 | `bzip2 -9` | 1,891,182 |
| xz | `xz -9` | 1,514,324 |
| xz extreme | `xz -9e` | 1,511,444 |
| xz with 4-byte delta | `xz --delta=dist=4 --lzma2=preset=9e` | 1,538,052 |

The bundled xz file is 41.3% smaller than the original and smallest among
these measured byte-stream formats. It retains the FITS headers exactly;
the comparison does not establish a globally optimal codec.

Lossless FITS tiled `GZIP_2` with one full-image tile and quantization disabled
measured 1,500,480 bytes, but rewrites the FITS container. The bundled archive
uses 10,964 more bytes to preserve the complete original byte stream.

Reproduce the archive with:

```sh
xz -9e -c CometCIEF_test.fits > CometCIEF_test.fits.xz
```

SHA-256 of the **decompressed** file:

```text
b9459e7484cb60e72ac53b14a620eef6d73449a2d6bd1f99e061bda67d114496
```

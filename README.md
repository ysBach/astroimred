# astroimred

Manage and manipulate [FITS](https://en.wikipedia.org/wiki/FITS) images.

[Documentation](https://ysbach.github.io/astroimred/) ·
[Build and update the docs](docs/README.md)

Recommended import:

```python
import astroimred as air
```

The initial code comes from legacy FITS utilities. Reduction utilities from
the former `fitsimred` package are being ported into `astroimred.reduction`.

## Documentation

The Quarto site combines executable FITS/photometry tutorials with API pages
generated from docstrings. Pushes to `main` update GitHub Pages after the docs
build passes; pull requests build without deploying.

Enable GitHub Actions as the Pages source once, following
[the setup guide](docs/README.md). Guides and docstrings follow the
[clear, complete writing convention](docs/README.md#authoring-convention).

## Releases

Run `release.yml` manually to rehearse a release. Push a matching `vX.Y.Z` tag
to publish tested wheels and a source distribution to PyPI, then create the
GitHub release. Manual runs never publish, including runs on tags.

Follow [RELEASING.md](RELEASING.md) for the one-time publisher setup, version
checks, and recovery steps. `astroimred` is a pure-Python distribution; its Rust
dependencies have their own crate releases.

# RawLoader

This Rust script is a minimalistic reference implementation of a RAW image loader used in my **RapidRAW** project. It supports many RAW formats (DNG, ARW, CR2, etc.) via the `rawloader` crate and applies a simple linear demosaicing algorithm.

## Features
- Decodes RAW files using [`rawloader`](https://crates.io/crates/rawloader)
- Supports many RAW formats (e.g. `.dng`, `.arw`, `.cr2`, `.nef`, etc.)
- Handles different Bayer patterns: RGGB, BGGR, GRBG, GBRG
- Performs white balancing, exposure correction, and gamma conversion
- Uses Rayon for parallel processing of image rows

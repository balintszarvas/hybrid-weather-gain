# Third-Party Code and Attribution

This repository builds on third-party open-source projects, especially Google Research weather/climate modeling code.

## Upstream Projects

- **Dinosaur**  
  Repository: https://github.com/google-research/dinosaur  
  License: Apache License 2.0

- **NeuralGCM**  
  Repository: https://github.com/google-research/neuralgcm  
  License: Apache License 2.0

## How Third-Party Code Is Used Here

- This repository imports and uses APIs from the projects above.
- Some implementation ideas and helper logic are adapted from those projects.
- Local folder `google_train/` may exist on the author's machine as a development mirror, but it is excluded from this repository's published snapshot.

## Attribution for Adapted Code

Where files include adapted logic, they include provenance comments linking to the upstream project and stating that modifications were made.

## Licensing Notes

- The original upstream code remains under its original license (Apache-2.0).
- This repository's original code is MIT-licensed (see `LICENSE`).
- No statement in this repository changes the license terms of third-party code.

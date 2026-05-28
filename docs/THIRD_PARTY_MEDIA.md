# Third-Party Media

This project uses small preview frames from the public DimOS repository only as platform context in the showcase video.

Source repository: https://github.com/dimensionalOS/dimos

License: the checked-in `LICENSE` file in `dimensionalOS/dimos` is Apache License 2.0 with copyright notice for Dimensional Inc. GitHub currently reports the repository license as `Other`, so the source license text should be treated as the controlling reference.

## Included Preview Frames

The raw GIF/MOV files are not vendored in this repository. They can be reproduced from the public DimOS repository with Git LFS:

```bash
git clone https://github.com/dimensionalOS/dimos.git
cd dimos
git lfs pull --include="assets/readme/navigation.gif,assets/readme/agentic_control.gif,assets/readme/spatial_memory.gif,assets/dimos_interface.gif,assets/trimmed_video_office.mov,docs/capabilities/navigation/native/assets/noros_nav.gif" --exclude=""
```

Preview frames were extracted from:

- `assets/readme/navigation.gif`
- `assets/readme/agentic_control.gif`
- `assets/readme/spatial_memory.gif`
- `assets/dimos_interface.gif`
- `assets/trimmed_video_office.mov`
- `docs/capabilities/navigation/native/assets/noros_nav.gif`

## Claim Boundary

These assets show public DimOS platform capabilities: navigation, agentic control, spatial memory, interface, and replay examples.

They are not presented as our robot run, not used as training data, and not used as evidence that our model controlled a robot. Our own hackathon contribution remains:

```text
real Go2 venue frames
-> label-safe counterfactual decision traces
-> micro world scorer
-> WorldForge-style evidence artifacts
```


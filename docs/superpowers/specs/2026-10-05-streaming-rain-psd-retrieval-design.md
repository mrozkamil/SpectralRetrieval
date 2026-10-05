# Streaming Rain PSD Retrieval Design

## Goal and scope

Build a maintainable rain particle-size-distribution retrieval that can process a multi-hour observation period with memory use bounded by a profile and compact temporal state. The first implementation is a runnable Python reference on the bundled observations; the production implementation is Julia, using automatic differentiation and performance-oriented profile processing. The retrieval covers each profile from the lowest usable radar gate to the melting-layer base and estimates a time sequence of rain and cloud microphysics and Doppler nuisance parameters.

The first physical process is rain-drop sedimentation. The design leaves a process interface for future microphysics but does not add other processes now. The Python prototype is a scientific reference and data-validation path, not the production performance target.

## Agreed inputs and parameterization

- Read radar spectra from whichever single, dual, or triple frequencies are present at a time. W band is supported when it is among the measurements; the implementation must not assume a fixed frequency count, P09 mode, or four W-band chirps.
- Accept the bundled `.znc` files as NetCDF/HDF5 regardless of their filename extension. Discover time, range, Doppler velocity, chirp, and instrument metadata from each file rather than hard-coding the sample file layout.
- Align profiles in time and use the W-band grid as the common reference grid when W band is available. Keep per-instrument/per-chirp masks and mappings so channels with different valid ranges, velocity bins, or missing samples remain representable. Without W band, construct the common grid from the available measurements.
- Represent rain PSD on a configurable logarithmic diameter grid from 0.1 mm to 8 mm. Eight millimetres is the initial upper limit; the weakly constrained large-drop tail must be identified in diagnostics. Retrieve spline coefficients in `log10(m^-4)` as deviations around a constant `N(D) = 1e-6 m^-4` background (`-60 dB(m^-4)`).
- Represent cloud droplets with a lognormal PSD. Retrieve number concentration and effective radius; hold geometric width at a configured value in the initial model so the cloud component has two unknowns per level.
- Retrieve vertical air motion, one turbulence parameter, and an instrument-specific noise floor at every gate. Estimate a noise prior from the Hildebrand-Sekhon method. Noise is added in linear power to the attenuated signal.
- Initialize the cloud and rain forward models from physically documented drop-size, fall-speed, and scattering tables. Use consistent units at module boundaries.

## State, objective, and temporal processing

At each time profile, the state contains rain spline coefficients, cloud number concentration and effective radius, vertical air motion, turbulence, and radar noise at each usable height (noise is separate for each measured frequency/channel where needed). The current profile's objective combines:

1. A likelihood for the observed Doppler spectra from every available radar frequency.
2. A lidar likelihood using a simulated attenuated-backscatter signal where valid lidar data are available.
3. A ground disdrometer likelihood on the predicted near-surface size-resolved rain flux/counts, with temporal integration and representativeness uncertainty represented explicitly.
4. A background penalty against the previous profile after it has been advanced by the sedimentation model over the sampling interval. Keep the rain birth/innovation uncertainty broad enough that a newly raining profile can be retrieved after a clear profile; the previous profile must not hard-zero new rain.
5. Parameter priors and regularization needed to keep the per-profile inversion stable.

Process the time sequence in two passes. The forward pass reads one profile at a time, retrieves its state, and propagates rain downward to form the next profile's background. It writes compact filtered-state and transition information to a checkpoint store. The backward pass reads those checkpoints in reverse time and propagates later-profile and ground-disdrometer information upward through the reverse-time conditional transition. This reverse pass is an information smoother for the forward sedimentation model, not an added physical process that makes drops rise. Raw multi-hour spectral data are not retained in RAM; memory use is bounded by the active profile, its optimizer workspace, and compact smoothing state. Checkpoints may scale with run length on disk.

The first implementation is a two-pass profile-wise smoother, not one dense optimization vector containing every height and time. It must still support processing an entire multi-hour run in one invocation. The smoother must allow information to traverse the full requested run; a future bounded-lag mode may be added as an operational option, but is not required for the initial design.

Rain sedimentation remaps each diameter class conservatively between heights using terminal fall speed over the profile sampling interval. The model must expose boundary inflow and process uncertainty so missing/clear prior rain does not prevent rain from entering the retrieved column. The ground observation constrains rain along its size-dependent fall-time path; for a 500 m path and 1 m/s drizzle, this is about 500 s or 125 profiles at the bundled W-band cadence of about 4 s.

## Forward operators and modular boundaries

- **Radar forward operator:** Given a profile state, compute size-conditioned Doppler spectra for cloud and rain, including terminal fall velocity, retrieved vertical air motion, and turbulence. Turbulence must affect the velocity distribution in a drop-size-dependent way, rather than convolving every size class with the same kernel. Add noise in linear power. Apply cumulative two-way attenuation at each radar frequency, accumulated from the instrument upward through every lower gate, to the simulated spectral power at that gate.
- **Attenuation constraint:** Retain the Rayleigh part of the spectrum in the likelihood because its frequency-dependent attenuation constrains differential attenuation. Compute attenuation for each height and propagate it to all higher gates in the profile.
- **Lidar forward operator:** Simulate attenuated lidar backscatter from cloud and drizzle using the instrument wavelength and available lidar metadata. Use valid attenuated-backscatter samples and Cloudnet category/mask information; do not treat the measured beta product as unattenuated backscatter.
- **Ground disdrometer operator:** Map the retrieved near-surface PSD and fall speeds to the disdrometer's size bins, sampling interval, and measured quantity (counts/flux). Carry a representativeness error because a point disdrometer sample and a radar volume do not observe identical space-time volumes.
- **Physical model interface:** Advance a rain PSD profile through one time interval by sedimentation and return the predicted next-profile rain background plus uncertainty. Other processes can be added behind this interface later.
- **Data and orchestration layer:** Read/align one profile at a time, build the channel-to-reference-grid mappings, call the forward operators and profile objective, persist compact checkpoints, and emit retrieved profiles and diagnostics.

Keep these modules independently callable so their physical calculations and interpolation/mapping can be validated separately. Use precomputed size/velocity mappings and spline basis matrices where they are state-independent; keep the differentiated objective allocation-light in Julia.

## Implementation path and performance

1. Restore a portable Python reference run by replacing absolute paths and undocumented fixed-file assumptions with explicit configuration and validation. Use the bundled observations to check the file reader, profile timing, channel metadata, and spectral forward/retrieval behavior that can be exercised with available data. Missing ancillary inputs must produce a precise diagnostic rather than an obscure import or file error.
2. Implement the modular retrieval in Julia with automatic differentiation for the per-profile objective and its derivatives. Preserve the same parameter definitions, physical units, objective terms, and reference cases as the Python implementation.
3. Keep input arrays profile-local; do not construct a full time-height-frequency-velocity cube. Persist compact per-profile checkpoints for the reverse pass. Benchmark the Python reference and Julia implementation on the same bundled case, reporting elapsed time, peak memory, and retrieval agreement. No fixed numerical speedup threshold has been specified yet.

## Initial acceptance criteria

- Bundled W-band `.znc` data are opened and their actual timestamps, cadence, chirps, ranges, and velocity grids are reported; the reader does not infer P09 or four chirps.
- The reader handles one, two, or three radar frequencies and profiles with missing channels, and uses W-grid reference mappings when W is present.
- A synthetic sedimentation case conserves rain number/flux to the documented boundary treatment, shifts each diameter class by its fall distance, and produces a clear-profile-to-rain transition when the current spectra support rain.
- The radar forward operator matches hand-computable attenuation and linear-noise cases, applies cumulative attenuation to every gate, and returns size-conditioned velocity kernels for multiple drop sizes.
- The cloud, lidar, and disdrometer operators each pass small synthetic reference cases with units and masks checked.
- A multi-profile synthetic period gives the same results whether loaded as one dataset or streamed profile by profile, within solver tolerance. The forward and reverse passes can process a multi-hour sequence without retaining raw spectra for the full period in RAM.
- Julia and Python reference outputs agree within documented tolerances on shared deterministic cases. Benchmarks report elapsed time and peak memory without imposing an unsupported speedup claim.

## Out of scope for the first implementation

- Ice/snow microphysics, coalescence, evaporation, breakup, and other new processes.
- Estimating cloud lognormal width, or adding more cloud PSD shape parameters.
- A dense joint batch optimizer over every profile in a multi-hour period.
- Using derived reflectivity as a second independent likelihood when it duplicates information already present in the Doppler spectra.
- Claims that the upper rain-size tail is well constrained merely because the grid extends to 8 mm.

## Scientific references

- Tridon et al. (2021), multifrequency Doppler-spectrum rain DSD retrieval: [Atmospheric Measurement Techniques 14, 511](https://amt.copernicus.org/articles/14/511/2021/).
- Turbulence effects on raindrop fall velocities: [Advances in Science and Research 18, 33 (2021)](https://asr.copernicus.org/articles/18/33/2021/).
- Triple-frequency differential-Doppler initial-guess context: [Earth and Space Science (2020)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019EA000789).

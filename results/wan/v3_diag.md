# VAE diagnostic: forcing latent equality on static positions (no transformer involved)

Each row decodes the full_30 reference latent after modifying only the 75% least-active positions in latent frames 2..8. Metrics vs the unmodified reference decode.

| prompt | variant | PSNR | LPIPS | flicker (ref mask) | motion | latent rel. change, static pos. |
|---|---|---|---|---|---|---|
| 00 | copy_frame1 | 24.95 | 0.142 | 1.47 (ref 0.05) | 1.64 | 0.114 |
| 00 | temporal_mean | 31.62 | 0.033 | 1.30 (ref 0.05) | 1.52 | 0.114 |
| 00 | untouched | 71.35 | 0.000 | 0.05 (ref 0.05) | 1.00 | 0.114 |
| 01 | copy_frame1 | 31.30 | 0.084 | 0.76 (ref 0.03) | 1.83 | 0.128 |
| 01 | temporal_mean | 39.92 | 0.009 | 0.19 (ref 0.03) | 1.18 | 0.128 |
| 01 | untouched | 75.15 | 0.000 | 0.03 (ref 0.03) | 1.00 | 0.128 |
| 02 | copy_frame1 | 27.59 | 0.055 | 0.36 (ref 0.00) | 3.43 | 0.111 |
| 02 | temporal_mean | 36.43 | 0.008 | 0.04 (ref 0.00) | 2.79 | 0.111 |
| 02 | untouched | 73.06 | 0.000 | 0.00 (ref 0.00) | 1.00 | 0.111 |
| 03 | copy_frame1 | 20.60 | 0.220 | 2.02 (ref 0.40) | 1.35 | 0.239 |
| 03 | temporal_mean | 24.96 | 0.140 | 0.71 (ref 0.40) | 0.96 | 0.239 |
| 03 | untouched | 71.54 | 0.000 | 0.40 (ref 0.40) | 1.00 | 0.239 |
| 04 | copy_frame1 | 25.82 | 0.063 | 1.00 (ref 0.08) | 1.30 | 0.149 |
| 04 | temporal_mean | 29.92 | 0.028 | 0.51 (ref 0.08) | 1.20 | 0.149 |
| 04 | untouched | 70.90 | 0.000 | 0.08 (ref 0.08) | 1.00 | 0.149 |
| 05 | copy_frame1 | 19.56 | 0.240 | 0.88 (ref 0.33) | 1.28 | 0.217 |
| 05 | temporal_mean | 23.44 | 0.319 | 0.68 (ref 0.33) | 1.06 | 0.217 |
| 05 | untouched | 65.58 | 0.000 | 0.33 (ref 0.33) | 1.00 | 0.217 |
| 06 | copy_frame1 | 21.77 | 0.127 | 1.00 (ref 0.37) | 1.03 | 0.281 |
| 06 | temporal_mean | 26.78 | 0.061 | 0.78 (ref 0.37) | 1.10 | 0.281 |
| 06 | untouched | 67.15 | 0.000 | 0.37 (ref 0.37) | 1.00 | 0.281 |
| 07 | copy_frame1 | 25.36 | 0.106 | 1.01 (ref 0.27) | 1.08 | 0.231 |
| 07 | temporal_mean | 30.07 | 0.102 | 0.53 (ref 0.27) | 0.93 | 0.231 |
| 07 | untouched | 69.23 | 0.000 | 0.27 (ref 0.27) | 1.00 | 0.231 |
| **mean** | copy_frame1 | 24.62 | 0.130 | 1.06 | 1.62 | 0.184 |
| **mean** | temporal_mean | 30.39 | 0.087 | 0.59 | 1.34 | 0.184 |
| **mean** | untouched | 70.49 | 0.000 | 0.19 | 1.00 | 0.184 |
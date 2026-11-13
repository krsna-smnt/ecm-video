# Kaggle Version 2: Wan 2.1 T2V 1.3B, 16 prompts (11 static camera, 5 moving camera)
33 frames, 480x832, Euler, shift 3, CFG 5. Reference = full_30, identical noise. (Pasted by KSG from
the Kaggle output results.md, 13 Nov; 16 of 24 prompts finished within the time budget.)

| method (all 16 prompts) | wall x | compute x | PSNR | SSIM | LPIPS | CLIP | flicker | motion | static | gated |
|---|---|---|---|---|---|---|---|---|---|---|
| full_30 | 1.00 | 1.00 | 99.00 | 1.000 | 0.000 | 29.96 | 0.56 | 1.00 | 0.00 | 0.00 |
| steps_20 | 1.50 | 1.50 | 19.15 | 0.748 | 0.254 | 30.01 | 0.82 | 0.92 | 0 | 0 |
| unipc_20 | 1.50 | 1.50 | 19.77 | 0.760 | 0.225 | 30.16 | 0.78 | 1.05 | 0 | 0 |
| steps_15 | 2.00 | 2.00 | 17.11 | 0.663 | 0.359 | 29.93 | 1.27 | 0.94 | 0 | 0 |
| unipc_15 | 2.00 | 2.00 | 17.82 | 0.687 | 0.308 | 30.62 | 1.04 | 1.09 | 0 | 0 |
| steps_10 | 3.00 | 3.00 | 15.92 | 0.626 | 0.426 | 29.92 | 1.36 | 0.84 | 0 | 0 |
| share_30_f0.75_warm | 1.42 | 1.38 | 40.02* | 0.724 | 0.272 | 29.51 | 1.41 | 1.55 | 0.56 | 0.25 |
| share_30_f0.5_draft3 | 1.26 | 1.41 | 12.33 | 0.391 | 0.664 | 25.88 | 4.45 | 2.57 | 0.50 | 0 |
| share_30_f0.75_draft3 | 2.08 | 1.94 | 12.23 | 0.403 | 0.761 | 28.58 | 5.73 | 3.92 | 0.75 | 0 |
| share_30_f0.9_draft3 | 2.98 | 2.50 | 12.31 | 0.406 | 0.777 | 28.89 | 5.48 | 3.80 | 0.90 | 0 |
| share_15_f0.75_draft3 | 3.43 | 3.24 | 12.19 | 0.409 | 0.756 | 29.01 | 6.05 | 4.33 | 0.75 | 0 |
| abl_noRenoise_30_f0.75_draft3 | 2.08 | 1.94 | 9.99 | 0.379 | 0.844 | 20.42 | 2.29 | 1.51 | 0.75 | 0 |
| abl_noBias_30_f0.75_draft3 | 2.67 | 1.94 | 10.52 | 0.328 | 0.712 | 26.72 | 10.50 | 8.33 | 0.75 | 0 |
| abl_noGate_30_f0.75_draft3 | 2.07 | 1.94 | 12.23 | 0.403 | 0.761 | 28.57 | 5.73 | 3.92 | 0.75 | 0 |

\* inflated: the pan gate fired on 4 of 16 prompts (3 of 11 static ones, all false alarms), and
those runs are identical to the reference (PSNR 99). On the 12 ungated prompts, PSNR ~20 and
LPIPS ~0.36, i.e. no better than steps_20 at similar speed.

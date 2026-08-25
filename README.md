# XiCon

Official repository for **"Contrastive learning for long-term time series forecasting with ξ-correlation"**  
M.S. thesis, Dept. of Statistics, Chung-Ang University (Aug 2024)  
Poster, The Korean Statistical Society Winter Conference, Nov 22–23 2024, Daejeon

## Overview

Contrastive learning for time series builds positive pairs from windows that are far apart in time.
AutoCon (Park et al., 2024) does this using **global autocorrelation** — but autocorrelation only
measures *linear* dependence between lagged values.

Real-world series often stay non-stationary even after differencing and log transformation, and in
that regime distant windows can be related **non-linearly**. Autocorrelation misses that relation
entirely.

XiCon adds Chatterjee's **ξ-correlation** (2021) — a rank-based, outlier-robust measure that detects
non-monotonic association — to the contrastive objective. Positive/negative pairs are selected by a
weighted combination of global autocorrelation and global ξ-correlation, controlled by `omega`.

The model architecture is unchanged from AutoCon (channel-independent, RevIN, TCN encoder,
multi-scale moving-average decoder). **Only the loss changes.**

## Method

For windows `w_i`, `w_j` drawn from different time points, we compute both the global
autocorrelation and the global ξ-correlation, then combine them:

```
R_ij = omega × |autocorr(i,j)| + (1 - omega) × xi_corr(i,j)
```

The pair with the highest `R_ij` becomes the positive; all pairs with lower `R_ij` become negatives.
`R_ij` is also used as a weight, so pairs with weaker association contribute less.

```
Loss = MSE + lambda × XiCon
```

- `omega = 1.0` → reduces to AutoCon
- `omega = 0.0` → pure ξ-correlation
- `omega = 0.3` was used for the main results

## Results

Validated on **9 benchmark datasets × 4 prediction horizons (36 intervals)**:
ETTm1, ETTm2, ETTh1, ETTh2, Traffic, Illness, Electricity, Weather, Exchange.
*The thesis (Aug 2024) reports results for six of these; Electricity, Weather, and Exchange were
added in follow-up experiments.*
Baselines: AutoCon, PatchTST, TimesNet, DLinear, Crossformer.

| Baseline | Avg. MSE improvement |
| -------- | -------------------- |
| AutoCon  | **5.81%**            |
| PatchTST | **8.52%**            |

The gain is **not uniform**, and where it appears is consistent with what the method does:

- **Gains grow with the strength of ξ-correlation in the data.** ETTm2 and ETTh2, whose variables
retain high (auto) ξ-correlation even at long lags, showed the largest improvement — over 10%
MSE reduction against AutoCon at some horizons.
- **Gains grow with prediction length.** The longer the horizon, the more the loss must rely on
dependence reaching beyond the look-back window.
- **No gain where ξ-correlation is absent.** On Illness, most variables fall below 0.2
ξ-correlation at long lags. XiCon gave almost no improvement there and TimesNet was the strongest
model overall. This is expected — if a series carries little non-linear lag dependence, the ξ term
has nothing to exploit.

**4 of 36 intervals underperform the baseline.** They are reported as-is rather than filtered out:
the method is a conditional improvement, not a universal one, and the condition is measurable in
advance by computing ξ-correlation on the target series before training.

## Environment setup

```
python == 3.8
torch == 1.7.1
numpy == 1.23.5
pandas
statsmodels
scikit-learn
einops
sympy
numba
```

## Run with command line

```
python -u run.py --XiCon --multiscales 96 --wnorm ReVIN --lambda 1.0 \
  --d_model 16 --d_ff 16 --e_layers 2 --target OT --c_out 1 \
  --root_path ./dataset/ETT-small --data_path ETTh1.csv \
  --model_id ICLR24_CRV --model XiCon --data ETTh1 \
  --seq_len 336 --label_len 48 --pred_len 96 --enc_in 1 \
  --des 'Exp' --itr 5 --batch_size 64 --learning_rate 0.01 \
  --feature S --omega 0.3
```

## Run with scripts

```
sh ./scripts/XiCon_{ETTh1|ETTh2|ETTm1|ETTm2|Electricity|Traffic|Weather|Exchange|Illness}.sh \
   {CUDA_VISIBLE_DEVICES} {NUM_RUNS}
```

Examples:

```
$ pwd
/home/user/XiCon

$ sh ./scripts/XiCon_ETTh2.sh 0 5
$ sh ./scripts/XiCon_Traffic.sh 0 5
```

## Reproducibility

Script naming convention:

| Suffix | Meaning |
| ------ | ------- |
| trailing number (e.g. `XiCon_Elec_s_revin_e-3`) | `lambda` tuning |
| `aa` (omega = 0.99) | almost AutoCon |
| `half` (omega = 0.5) | equal weight |
| `ax` (omega = 0.01) | almost pure XiCon |

The `aa` / `half` / `ax` triplet exists to isolate the contribution of the ξ term: `aa` reproduces
AutoCon behaviour, so the gap between `aa` and `ax` on the same dataset is attributable to
ξ-correlation rather than to any other change.

## Citation

```
임재승 (2024). ξ-상관을 이용한 장기 시계열 예측을 위한 대조학습.
석사학위논문, 중앙대학교 대학원 통계학과.

Yim, J. (2024). Contrastive learning for long-term time series forecasting
with ξ-correlation. Poster presented at the Korean Statistical Society
Winter Conference, Daejeon, Korea, November 22–23, 2024.
```

### References

- Chatterjee, S. (2021). A new coefficient of correlation. *JASA*, 116(536), 2009–2022.
- Park, J., Gwak, D., Choo, J., & Choi, E. (2024). Self-supervised contrastive forecasting. arXiv:2402.02023.
- Nie, Y. et al. (2023). A time series is worth 64 words: Long-term forecasting with Transformers. *ICLR*.
- Kim, T. et al. (2021). Reversible instance normalization for accurate time-series forecasting. *ICLR*.

# TopFD Run — new code, relaxed filtering (min match 2, max-miss 4, no mz-filter)

_Run timestamp: 2026-06-14 16:52:08_
_Branch / commit: `debug_mms` @ `4c9d4f1`_

## Code changes used in this run

- `EnvPara::min_match_peak_num_` and `min_consecutive_peak_num_` for the
  high-mass group lowered from 3 to 2 (`{1,2,3}` → `{1,2,2}`).
- New hidden options: `--max-miss-peak-num`, `--disable-filter-by-mz`,
  `--output-dp-envs`.

## Parameters

| Flag | Meaning | Value |
|---|---|---|
| `-T` | text peak-list input (single spectrum) | on |
| `--gene-sql` | write deconvoluted spectrum to SQLite | on |
| `-c 3` | max charge state | 3 |
| `-m 1620` | max monoisotopic mass (Da) | 1620 |
| `-s 0.1` | MS/MS signal-to-noise ratio | 0.1 |
| `-d` | disable fragment-number filtering | on |
| `-t 0` | ECScore cutoff | 0 |
| `-e 0.04` | m/z error tolerance | 0.04 |
| `--max-miss-peak-num 4` | max missing peaks in a matched envelope | 4 |
| `--disable-filter-by-mz` | skip the filtering-by-mz step | on |
| `--output-dp-envs` | dump candidate/filtered/window/DP envelopes | on |

## Command

```bash
/media/xiaowen/data/code/toppic_claude/bin/topfd \
  -T --gene-sql -c 3 -m 1620 -s 0.1 -d -t 0 -e 0.04 \
  --output-dp-envs --max-miss-peak-num 4 --disable-filter-by-mz \
  peak_list_topfd.txt
```

(Run from `exp_minmatch2/`, which holds a copy of `peak_list_topfd.txt`.)

## Peak counts

| File | Peaks |
|---|---|
| `peak_list_topfd.txt` (input) | 174 |
| `deconv_ms2.env` (deconvoluted peak rows, excl. header) | 127 |
| `deconv_ms2.msalign` (deconvoluted masses) | 51 |

Deconvolution assigned **127** of the 174 raw peaks to deconvoluted envelopes
(the peak rows in `deconv_ms2.env`; 121 distinct experimental peaks, a few
shared across overlapping envelopes), and reported **51** deconvoluted
monoisotopic masses. SQLite cross-check: `ms2_peak` = 174, `ms2_env` = 51,
`ms2_env_peak` = 213.

## Output

In `exp_minmatch2/`: `deconv_ms2.msalign`, `deconv_ms2.env`, `deconv.sqlite`,
plus the `--output-dp-envs` dumps (`cand_envs.txt`, `filtered_envs.txt`,
`win_envs.txt`, `dp_envs.txt`) and `run.log`.

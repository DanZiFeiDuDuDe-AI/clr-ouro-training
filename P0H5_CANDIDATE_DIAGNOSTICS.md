# P0H5_CANDIDATE_DIAGNOSTICS.md

Date: 2026-08-25
N=150 (same draw: seed 42, after 600 replay)

## Per-model summary

| Model | acc | rank=1 | rank=2 | rank=3 | rank>3 | mean rank | mean margin | mean entropy | mean Spearman(vs M0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_teacher | 11/150 | 11 | 17 | 20 | 102 | 4.47 | -2.0516 | 1.464 | 1.000 |
| M1 | 15/150 | 15 | 7 | 0 | 128 | 10.07 | -9.9146 | 1.076 | 0.699 |
| norm600 | 7/150 | 7 | 2 | 4 | 137 | 9.59 | -4.7279 | 1.835 | 0.775 |
| norm1800 | 0/150 | 0 | 0 | 4 | 146 | 9.85 | -4.9246 | 1.632 | 0.743 |

## Per-question detail (first 20 per model)

| Model | traj | correct | pred | rank | margin | entropy | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0_teacher | 0 | brother | nephew | 4 | -2.5874 | 1.195 | 1.000 |
| M0_teacher | 1 | brother | nephew | 4 | -2.2640 | 1.266 | 1.000 |
| M0_teacher | 2 | mother | nephew | 6 | -2.4831 | 1.570 | 1.000 |
| M0_teacher | 3 | mother | nephew | 6 | -2.5927 | 1.495 | 1.000 |
| M0_teacher | 4 | mother | nephew | 4 | -1.9174 | 1.582 | 1.000 |
| M0_teacher | 5 | sister | sister | 1 | +0.1924 | 1.356 | 1.000 |
| M0_teacher | 6 | son | nephew | 5 | -2.2244 | 1.648 | 1.000 |
| M0_teacher | 7 | son | nephew | 4 | -2.1485 | 1.464 | 1.000 |
| M0_teacher | 8 | son | nephew | 6 | -3.3845 | 1.121 | 1.000 |
| M0_teacher | 9 | mother | nephew | 4 | -2.1632 | 1.533 | 1.000 |
| M0_teacher | 10 | son | nephew | 6 | -2.0884 | 1.768 | 1.000 |
| M0_teacher | 11 | daughter | nephew | 2 | -0.8927 | 1.783 | 1.000 |
| M0_teacher | 12 | son | nephew | 6 | -1.7885 | 1.828 | 1.000 |
| M0_teacher | 13 | sister | nephew | 2 | -0.5260 | 1.225 | 1.000 |
| M0_teacher | 14 | father | grandfather | 6 | -2.2630 | 1.644 | 1.000 |
| M0_teacher | 15 | father | nephew | 4 | -1.6451 | 1.681 | 1.000 |
| M0_teacher | 16 | son | nephew | 9 | -3.8782 | 1.467 | 1.000 |
| M0_teacher | 17 | brother | nephew | 3 | -1.0985 | 1.573 | 1.000 |
| M0_teacher | 18 | father | nephew | 4 | -1.3567 | 1.777 | 1.000 |
| M0_teacher | 19 | brother | nephew | 3 | -2.0918 | 1.200 | 1.000 |
| M1 | 0 | brother | sister | 12 | -12.8591 | 1.044 | 0.732 |
| M1 | 1 | brother | nephew | 13 | -12.4986 | 1.055 | 0.697 |
| M1 | 2 | mother | sister | 13 | -13.6807 | 1.145 | 0.653 |
| M1 | 3 | mother | nephew | 11 | -12.5137 | 0.731 | 0.670 |
| M1 | 4 | mother | nephew | 11 | -11.9927 | 0.630 | 0.596 |
| M1 | 5 | sister | sister | 1 | +0.2322 | 1.270 | 0.798 |
| M1 | 6 | son | sister | 11 | -10.9109 | 1.181 | 0.692 |
| M1 | 7 | son | nephew | 9 | -10.0782 | 1.203 | 0.793 |
| M1 | 8 | son | sister | 9 | -10.0653 | 1.148 | 0.807 |
| M1 | 9 | mother | nephew | 13 | -14.1471 | 1.073 | 0.556 |
| M1 | 10 | son | nephew | 12 | -10.9667 | 0.762 | 0.688 |
| M1 | 11 | daughter | nephew | 14 | -14.1906 | 0.965 | 0.429 |
| M1 | 12 | son | nephew | 9 | -8.6577 | 1.085 | 0.824 |
| M1 | 13 | sister | sister | 1 | +1.6815 | 0.739 | 0.859 |
| M1 | 14 | father | nephew | 10 | -9.9514 | 1.172 | 0.684 |
| M1 | 15 | father | nephew | 10 | -9.6885 | 0.935 | 0.736 |
| M1 | 16 | son | sister | 10 | -10.7533 | 0.683 | 0.785 |
| M1 | 17 | brother | nephew | 14 | -11.9816 | 1.228 | 0.635 |
| M1 | 18 | father | nephew | 10 | -8.6125 | 1.051 | 0.727 |
| M1 | 19 | brother | sister | 13 | -12.3207 | 1.232 | 0.675 |
| norm600 | 0 | brother | uncle | 12 | -4.9746 | 1.941 | 0.723 |
| norm600 | 1 | brother | nephew | 9 | -5.7947 | 1.800 | 0.916 |
| norm600 | 2 | mother | uncle | 9 | -3.4117 | 2.026 | 0.855 |
| norm600 | 3 | mother | nephew | 9 | -4.4767 | 1.816 | 0.864 |
| norm600 | 4 | mother | nephew | 9 | -4.3774 | 1.863 | 0.820 |
| norm600 | 5 | sister | uncle | 2 | -0.0483 | 1.956 | 0.837 |
| norm600 | 6 | son | nephew | 11 | -4.8858 | 1.903 | 0.811 |
| norm600 | 7 | son | nephew | 10 | -5.7189 | 1.827 | 0.811 |
| norm600 | 8 | son | uncle | 10 | -5.0809 | 1.952 | 0.824 |
| norm600 | 9 | mother | grandmother | 9 | -4.0642 | 1.993 | 0.833 |
| norm600 | 10 | son | nephew | 11 | -5.1478 | 1.623 | 0.842 |
| norm600 | 11 | daughter | uncle | 14 | -6.3758 | 1.939 | 0.429 |
| norm600 | 12 | son | sister | 12 | -5.5516 | 1.702 | 0.829 |
| norm600 | 13 | sister | nephew | 4 | -0.6635 | 1.941 | 0.798 |
| norm600 | 14 | father | uncle | 10 | -4.6565 | 1.864 | 0.846 |
| norm600 | 15 | father | uncle | 10 | -5.2653 | 1.969 | 0.820 |
| norm600 | 16 | son | nephew | 10 | -4.8828 | 1.861 | 0.921 |
| norm600 | 17 | brother | sister | 13 | -7.1417 | 1.887 | 0.657 |
| norm600 | 18 | father | uncle | 10 | -4.8692 | 1.810 | 0.815 |
| norm600 | 19 | brother | uncle | 11 | -6.1042 | 1.659 | 0.780 |
| norm1800 | 0 | brother | uncle | 13 | -6.9346 | 1.329 | 0.587 |
| norm1800 | 1 | brother | uncle | 13 | -5.9273 | 1.668 | 0.600 |
| norm1800 | 2 | mother | uncle | 10 | -5.4646 | 1.649 | 0.745 |
| norm1800 | 3 | mother | uncle | 10 | -5.2752 | 1.441 | 0.802 |
| norm1800 | 4 | mother | uncle | 9 | -4.2027 | 1.821 | 0.749 |
| norm1800 | 5 | sister | uncle | 6 | -1.8700 | 1.670 | 0.714 |
| norm1800 | 6 | son | uncle | 9 | -4.3265 | 1.807 | 0.833 |
| norm1800 | 7 | son | uncle | 9 | -4.7529 | 1.550 | 0.859 |
| norm1800 | 8 | son | uncle | 9 | -4.3943 | 1.746 | 0.802 |
| norm1800 | 9 | mother | uncle | 10 | -4.6721 | 1.702 | 0.736 |
| norm1800 | 10 | son | uncle | 7 | -4.1492 | 1.431 | 0.934 |
| norm1800 | 11 | daughter | uncle | 12 | -6.3081 | 1.716 | 0.525 |
| norm1800 | 12 | son | uncle | 8 | -3.6889 | 1.721 | 0.864 |
| norm1800 | 13 | sister | uncle | 7 | -2.3559 | 1.823 | 0.670 |
| norm1800 | 14 | father | uncle | 12 | -6.2122 | 1.558 | 0.780 |
| norm1800 | 15 | father | uncle | 11 | -5.6929 | 1.804 | 0.763 |
| norm1800 | 16 | son | uncle | 7 | -3.1761 | 1.665 | 0.811 |
| norm1800 | 17 | brother | uncle | 13 | -7.2224 | 1.668 | 0.499 |
| norm1800 | 18 | father | uncle | 11 | -5.6714 | 1.763 | 0.758 |
| norm1800 | 19 | brother | uncle | 12 | -6.3403 | 1.580 | 0.754 |

## Interpretation

norm1800 (d3): rank2+3=4, rank>3=146, entropy=1.632 (M0=1.464), Spearman=0.743

- 未落入明显单类死法，需结合 rank 分布细看

## Verdict gate

> **Systematic candidate-ranking degradation confirmed — leaning H2.**
> - norm1800 correct-candidate rank: rank2+3=4, rank>3=146 (97.3%), mean_rank=9.85 vs M0=4.47.
> - Not the classic "差一点点" failure (rank2+3 ≈ 2.7% only), NOT entropy collapse
>   (1.632 vs M0 1.464, +11%), NOT pure Spearman collapse (0.743, moderate).
> - The dominant failure is a systematic drop of correct-candidate rank toward
>   below-random (random ≈ 7.5; norm1800 ≈ 9.85), with margin shifting more negative
>   (-4.92 vs M0 -2.05): distillation improves logit/hidden fidelity while degrading
>   candidate behavior — matches the predicted "distillation 越深、candidate ranking 越差" pattern.
> - Therefore: DO NOT freeze norm1800 as M2_CANONICAL for 0I.
>   Proceed to H2 rank-4 LoRA on Q/K/V/O of the 6 converted layers (low LR,
>   objective priority: candidate behavior > NLL/logit fidelity > hidden cosine).
> - Keep norm1800 checkpoint untouched; do not overwrite. (Confirm with 网友 before training.)

## Raw rows (all 150)

```json
{
 "M0_teacher": [
  {
   "traj": 0,
   "rank": 4,
   "margin": -2.587445619205634,
   "entropy": 1.194919104387572,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 1,
   "rank": 4,
   "margin": -2.263991490006447,
   "entropy": 1.2663336188266752,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 2,
   "rank": 6,
   "margin": -2.4831348657608032,
   "entropy": 1.570358495689204,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 3,
   "rank": 6,
   "margin": -2.592711478471756,
   "entropy": 1.494830565297441,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 4,
   "rank": 4,
   "margin": -1.917393058538437,
   "entropy": 1.5818885655244077,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 5,
   "rank": 1,
   "margin": 0.1923674891392393,
   "entropy": 1.3562670122125213,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 6,
   "rank": 5,
   "margin": -2.224419491986434,
   "entropy": 1.6477289968216915,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 7,
   "rank": 4,
   "margin": -2.148540571331978,
   "entropy": 1.4636318259424155,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 8,
   "rank": 6,
   "margin": -3.384521109362443,
   "entropy": 1.1206827722436699,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 9,
   "rank": 4,
   "margin": -2.16318757335345,
   "entropy": 1.532516098214545,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 10,
   "rank": 6,
   "margin": -2.088404417037964,
   "entropy": 1.7677482126394335,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 11,
   "rank": 2,
   "margin": -0.8927357395490008,
   "entropy": 1.7833910206362626,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 12,
   "rank": 6,
   "margin": -1.7885071585575743,
   "entropy": 1.8276274449646468,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 13,
   "rank": 2,
   "margin": -0.5260467578967409,
   "entropy": 1.2253642626832784,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 14,
   "rank": 6,
   "margin": -2.263031207025051,
   "entropy": 1.6444415029701192,
   "spearman": 1.0,
   "correct": "father",
   "pred": "grandfather"
  },
  {
   "traj": 15,
   "rank": 4,
   "margin": -1.6451106518507004,
   "entropy": 1.68117120451308,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 16,
   "rank": 9,
   "margin": -3.8782286917169886,
   "entropy": 1.4674760866144687,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 17,
   "rank": 3,
   "margin": -1.098532622059186,
   "entropy": 1.5734693533979163,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 18,
   "rank": 4,
   "margin": -1.356695731480916,
   "entropy": 1.7770193538133199,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 19,
   "rank": 3,
   "margin": -2.091833519438903,
   "entropy": 1.2001203088337087,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 20,
   "rank": 3,
   "margin": -2.4152967631816864,
   "entropy": 1.2053898128084235,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 21,
   "rank": 4,
   "margin": -2.4570777515570326,
   "entropy": 1.5408368304663438,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 22,
   "rank": 6,
   "margin": -1.9737162739038467,
   "entropy": 1.787066138450362,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 23,
   "rank": 5,
   "margin": -2.643608455856641,
   "entropy": 1.4507381792279739,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 24,
   "rank": 2,
   "margin": -0.6866379529237747,
   "entropy": 1.2421273864514897,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 25,
   "rank": 4,
   "margin": -2.496849762896697,
   "entropy": 1.2559754892800477,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 26,
   "rank": 4,
   "margin": -2.103264406323433,
   "entropy": 1.3233495406137326,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 27,
   "rank": 4,
   "margin": -2.4036226347088814,
   "entropy": 1.4152088984687463,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 28,
   "rank": 4,
   "margin": -2.148540571331978,
   "entropy": 1.4636318259424155,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 29,
   "rank": 7,
   "margin": -2.8741543988386793,
   "entropy": 1.561611681837729,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 30,
   "rank": 1,
   "margin": 0.18798537304004004,
   "entropy": 1.3314122224620408,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 31,
   "rank": 2,
   "margin": -1.2474758128325147,
   "entropy": 1.5622264195361213,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 32,
   "rank": 6,
   "margin": -2.739351178208987,
   "entropy": 1.6564642944088135,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 33,
   "rank": 5,
   "margin": -2.6109499012430506,
   "entropy": 1.3987974319922323,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 34,
   "rank": 1,
   "margin": 0.4560975655913353,
   "entropy": 1.1003282866465094,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 35,
   "rank": 9,
   "margin": -3.8782286917169886,
   "entropy": 1.4674760866144687,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 36,
   "rank": 5,
   "margin": -2.5534288858373957,
   "entropy": 1.4786405958892055,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 37,
   "rank": 4,
   "margin": -2.787137006719907,
   "entropy": 1.2426515392357362,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 38,
   "rank": 9,
   "margin": -3.205418186883132,
   "entropy": 1.4506971407024112,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 39,
   "rank": 5,
   "margin": -2.7392857248584432,
   "entropy": 1.3188543258085907,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 40,
   "rank": 7,
   "margin": -3.1169186780850096,
   "entropy": 1.495726322305948,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 41,
   "rank": 6,
   "margin": -2.63515264292558,
   "entropy": 1.5456030620141694,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 42,
   "rank": 6,
   "margin": -2.6998646830519037,
   "entropy": 1.4200444409938735,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 43,
   "rank": 5,
   "margin": -1.1652390137314796,
   "entropy": 1.8161962294899219,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 44,
   "rank": 8,
   "margin": -2.940375042458375,
   "entropy": 1.4972369005281823,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 45,
   "rank": 7,
   "margin": -2.8741543988386793,
   "entropy": 1.561611681837729,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 46,
   "rank": 7,
   "margin": -3.1169186780850096,
   "entropy": 1.495726322305948,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 47,
   "rank": 6,
   "margin": -3.265090097983678,
   "entropy": 1.3835676630532685,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 48,
   "rank": 4,
   "margin": -2.844056611259778,
   "entropy": 1.226466474023938,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 49,
   "rank": 4,
   "margin": -1.5824904590845108,
   "entropy": 1.65759017327887,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 50,
   "rank": 8,
   "margin": -3.6099918981393175,
   "entropy": 1.4631523038823993,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 51,
   "rank": 4,
   "margin": -2.2671005576848984,
   "entropy": 1.3819619042627036,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 52,
   "rank": 3,
   "margin": -1.6801625937223434,
   "entropy": 1.420457815098597,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 53,
   "rank": 2,
   "margin": -0.8927357395490008,
   "entropy": 1.7833910206362626,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 54,
   "rank": 4,
   "margin": -2.3361683065692587,
   "entropy": 1.3254140825572989,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 55,
   "rank": 3,
   "margin": -2.0628814200560255,
   "entropy": 1.5970835930164622,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 56,
   "rank": 4,
   "margin": -2.1262717694044113,
   "entropy": 1.5317507385076554,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 57,
   "rank": 3,
   "margin": -1.0726960251728697,
   "entropy": 1.7922857309587017,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 58,
   "rank": 1,
   "margin": 0.6300459951162338,
   "entropy": 1.0188252404611815,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 59,
   "rank": 6,
   "margin": -2.3552657191952067,
   "entropy": 1.6010758083838987,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 60,
   "rank": 2,
   "margin": -0.5089291383822756,
   "entropy": 1.2730667968767366,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 61,
   "rank": 9,
   "margin": -3.205418186883132,
   "entropy": 1.4506971407024112,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 62,
   "rank": 3,
   "margin": -2.0493357976277666,
   "entropy": 1.2689803347105573,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 63,
   "rank": 2,
   "margin": -0.2553259159127874,
   "entropy": 1.1978180248675654,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 64,
   "rank": 3,
   "margin": -1.9322960947950678,
   "entropy": 1.6211225400131954,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 65,
   "rank": 7,
   "margin": -3.1169186780850096,
   "entropy": 1.495726322305948,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 66,
   "rank": 5,
   "margin": -1.5205926448106766,
   "entropy": 1.8503239417919826,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 67,
   "rank": 4,
   "margin": -2.471918563048045,
   "entropy": 1.4795072999321193,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 68,
   "rank": 7,
   "margin": -3.0300142367680865,
   "entropy": 1.65789163581616,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 69,
   "rank": 4,
   "margin": -2.7968247781197233,
   "entropy": 1.2471079149327369,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 70,
   "rank": 3,
   "margin": -1.8414334903160734,
   "entropy": 1.2915355634362577,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 71,
   "rank": 5,
   "margin": -2.7828804180026054,
   "entropy": 1.3709203871574007,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 72,
   "rank": 2,
   "margin": -0.630706898868084,
   "entropy": 1.2673590355789757,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 73,
   "rank": 7,
   "margin": -3.2452874531348543,
   "entropy": 1.5111877299055918,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 74,
   "rank": 5,
   "margin": -2.7463974853356676,
   "entropy": 1.4889242297351268,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 75,
   "rank": 3,
   "margin": -2.042660380403201,
   "entropy": 1.370761579455837,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 76,
   "rank": 3,
   "margin": -1.5950852582852049,
   "entropy": 1.5306585091294176,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 77,
   "rank": 6,
   "margin": -2.622567154467106,
   "entropy": 1.5945262578581516,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 78,
   "rank": 2,
   "margin": -0.5730850448211031,
   "entropy": 1.3918975734090948,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 79,
   "rank": 5,
   "margin": -2.9675200531880064,
   "entropy": 1.4818901630435415,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 80,
   "rank": 1,
   "margin": 0.4377106105287867,
   "entropy": 1.088815563384349,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 81,
   "rank": 7,
   "margin": -3.0406863192717237,
   "entropy": 1.6034246739030513,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 82,
   "rank": 2,
   "margin": -0.8633644680182142,
   "entropy": 1.5457148183004341,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 83,
   "rank": 6,
   "margin": -2.904781935115655,
   "entropy": 1.597810276650162,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 84,
   "rank": 2,
   "margin": -0.19941375652949045,
   "entropy": 1.331591099740808,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 85,
   "rank": 6,
   "margin": -2.5441261430581408,
   "entropy": 1.5912392912280862,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 86,
   "rank": 9,
   "margin": -5.115743892888228,
   "entropy": 1.1772415807844387,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 87,
   "rank": 3,
   "margin": -1.5440210302670794,
   "entropy": 1.4327579317052217,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 88,
   "rank": 7,
   "margin": -2.983956788976987,
   "entropy": 1.5708810851513944,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 89,
   "rank": 7,
   "margin": -3.579693853855133,
   "entropy": 1.5649700119080718,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 90,
   "rank": 4,
   "margin": -1.6746487220128374,
   "entropy": 1.585265610317972,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 91,
   "rank": 5,
   "margin": -1.948066532611847,
   "entropy": 1.6459538993049263,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 92,
   "rank": 4,
   "margin": -1.6451106518507004,
   "entropy": 1.68117120451308,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 93,
   "rank": 4,
   "margin": -1.2130677551031113,
   "entropy": 1.8820701871874468,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 94,
   "rank": 4,
   "margin": -2.551899453004201,
   "entropy": 1.1163444199459434,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 95,
   "rank": 7,
   "margin": -3.3505304853121443,
   "entropy": 1.4144970605622258,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 96,
   "rank": 2,
   "margin": -0.49163038035233786,
   "entropy": 1.2929000294928152,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 97,
   "rank": 9,
   "margin": -5.0270194709300995,
   "entropy": 1.2474078219942166,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 98,
   "rank": 9,
   "margin": -5.000845710436503,
   "entropy": 1.178971386851756,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 99,
   "rank": 3,
   "margin": -1.4648392746845884,
   "entropy": 1.4395686917913453,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 100,
   "rank": 4,
   "margin": -2.8195373614629107,
   "entropy": 1.1807518111557633,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 101,
   "rank": 5,
   "margin": -2.0122081339359283,
   "entropy": 1.6480049056121624,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 102,
   "rank": 6,
   "margin": -2.7132457097371416,
   "entropy": 1.5407343164191079,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 103,
   "rank": 6,
   "margin": -2.6887415250142412,
   "entropy": 1.5172975028447535,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 104,
   "rank": 5,
   "margin": -1.6956381350755692,
   "entropy": 1.7568303997618488,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 105,
   "rank": 7,
   "margin": -3.1551524450381594,
   "entropy": 1.6071034136605056,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 106,
   "rank": 8,
   "margin": -3.017805516719818,
   "entropy": 1.4783034912518813,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 107,
   "rank": 4,
   "margin": -2.71536373347044,
   "entropy": 1.372371563929848,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 108,
   "rank": 2,
   "margin": -0.21164229015509317,
   "entropy": 1.22511283571952,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 109,
   "rank": 6,
   "margin": -2.384456877907117,
   "entropy": 1.6467299654051475,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 110,
   "rank": 5,
   "margin": -2.7392857248584432,
   "entropy": 1.3188543258085907,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 111,
   "rank": 3,
   "margin": -1.978630582491557,
   "entropy": 1.2832302969195997,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 112,
   "rank": 1,
   "margin": 0.2221912145614624,
   "entropy": 1.3073039615391608,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 113,
   "rank": 4,
   "margin": -2.4570777515570326,
   "entropy": 1.5408368304663438,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 114,
   "rank": 2,
   "margin": -0.5730850448211031,
   "entropy": 1.3918975734090948,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 115,
   "rank": 4,
   "margin": -2.087940076986949,
   "entropy": 1.3442863987602895,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 116,
   "rank": 4,
   "margin": -1.563048630952835,
   "entropy": 1.6287739024440198,
   "spearman": 1.0,
   "correct": "father",
   "pred": "grandfather"
  },
  {
   "traj": 117,
   "rank": 2,
   "margin": -0.8240929817159968,
   "entropy": 1.6114795791323924,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 118,
   "rank": 7,
   "margin": -2.9695476094881696,
   "entropy": 1.555721555017654,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 119,
   "rank": 5,
   "margin": -2.134060690800349,
   "entropy": 1.6690170681774281,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 120,
   "rank": 6,
   "margin": -2.622567154467106,
   "entropy": 1.5945262578581516,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 121,
   "rank": 3,
   "margin": -1.9494742775956793,
   "entropy": 1.5113720491951004,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 122,
   "rank": 4,
   "margin": -2.714532122015953,
   "entropy": 1.2321980037150269,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 123,
   "rank": 4,
   "margin": -2.45000842710336,
   "entropy": 1.2532559670286263,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 124,
   "rank": 6,
   "margin": -1.5029734646280604,
   "entropy": 1.8811051797643692,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 125,
   "rank": 4,
   "margin": -1.1993366479873657,
   "entropy": 1.7425572679450905,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 126,
   "rank": 3,
   "margin": -1.877055548131466,
   "entropy": 1.4058711834117184,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 127,
   "rank": 3,
   "margin": -1.0726960251728697,
   "entropy": 1.7922857309587017,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 128,
   "rank": 2,
   "margin": -0.9966496427853899,
   "entropy": 1.5331342736747569,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 129,
   "rank": 6,
   "margin": -2.5282547380775213,
   "entropy": 1.5762992268225493,
   "spearman": 1.0,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 130,
   "rank": 3,
   "margin": -2.4459261049826937,
   "entropy": 1.4322753346446946,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 131,
   "rank": 4,
   "margin": -1.8551606933275862,
   "entropy": 1.6480576338026938,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 132,
   "rank": 6,
   "margin": -2.7091251264015836,
   "entropy": 1.5733545481352484,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 133,
   "rank": 4,
   "margin": -1.402010415991147,
   "entropy": 1.5336231339841426,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 134,
   "rank": 2,
   "margin": -0.6866379529237747,
   "entropy": 1.2421273864514897,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 135,
   "rank": 3,
   "margin": -1.7034123589595156,
   "entropy": 1.608570613191861,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 136,
   "rank": 1,
   "margin": 0.8797551790873213,
   "entropy": 0.9928180576632424,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 137,
   "rank": 1,
   "margin": 0.18549431363741586,
   "entropy": 1.3280760611921087,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 138,
   "rank": 1,
   "margin": 0.10507977505524924,
   "entropy": 1.1023671357611957,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 139,
   "rank": 1,
   "margin": 0.06193811322251985,
   "entropy": 1.3076559573489672,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 140,
   "rank": 5,
   "margin": -2.5942018181085587,
   "entropy": 1.4844980950278144,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 141,
   "rank": 1,
   "margin": 0.23941301802794168,
   "entropy": 1.1389380437878145,
   "spearman": 1.0,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 142,
   "rank": 4,
   "margin": -1.7860590629279613,
   "entropy": 1.646127936763723,
   "spearman": 1.0,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 143,
   "rank": 6,
   "margin": -2.3649396250645323,
   "entropy": 1.671592031611961,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 144,
   "rank": 6,
   "margin": -3.384521109362443,
   "entropy": 1.1206827722436699,
   "spearman": 1.0,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 145,
   "rank": 8,
   "margin": -3.6334814329942065,
   "entropy": 1.559796348389201,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 146,
   "rank": 7,
   "margin": -3.1894119133551913,
   "entropy": 1.4741485205809157,
   "spearman": 1.0,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 147,
   "rank": 4,
   "margin": -2.9263412306706114,
   "entropy": 1.3846378824505297,
   "spearman": 1.0,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 148,
   "rank": 3,
   "margin": -1.5440210302670794,
   "entropy": 1.4327579317052217,
   "spearman": 1.0,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 149,
   "rank": 5,
   "margin": -2.380543117721875,
   "entropy": 1.4883901016644305,
   "spearman": 1.0,
   "correct": "daughter",
   "pred": "nephew"
  }
 ],
 "M1": [
  {
   "traj": 0,
   "rank": 12,
   "margin": -12.859092712402344,
   "entropy": 1.043513097750514,
   "spearman": 0.7318681318681318,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 1,
   "rank": 13,
   "margin": -12.498573780059814,
   "entropy": 1.0554710881187073,
   "spearman": 0.6967032967032967,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 2,
   "rank": 13,
   "margin": -13.6806960105896,
   "entropy": 1.1446554235150634,
   "spearman": 0.6527472527472528,
   "correct": "mother",
   "pred": "sister"
  },
  {
   "traj": 3,
   "rank": 11,
   "margin": -12.513708432515463,
   "entropy": 0.7308875635844497,
   "spearman": 0.6703296703296703,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 4,
   "rank": 11,
   "margin": -11.992703596750895,
   "entropy": 0.6296087178685146,
   "spearman": 0.5956043956043956,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 5,
   "rank": 1,
   "margin": 0.23220451672871967,
   "entropy": 1.2695571598292745,
   "spearman": 0.7978021978021979,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 6,
   "rank": 11,
   "margin": -10.910904884338379,
   "entropy": 1.1811845569424106,
   "spearman": 0.6923076923076923,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 7,
   "rank": 9,
   "margin": -10.078191121419271,
   "entropy": 1.2026857453400779,
   "spearman": 0.7934065934065934,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 8,
   "rank": 9,
   "margin": -10.065321922302246,
   "entropy": 1.1479709778465048,
   "spearman": 0.8065934065934066,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 9,
   "rank": 13,
   "margin": -14.147091070810953,
   "entropy": 1.0728357421493362,
   "spearman": 0.5560439560439561,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 10,
   "rank": 12,
   "margin": -10.966742833455404,
   "entropy": 0.7622935958880968,
   "spearman": 0.6879120879120879,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 11,
   "rank": 14,
   "margin": -14.190618197123209,
   "entropy": 0.9652590176194809,
   "spearman": 0.42857142857142855,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 12,
   "rank": 9,
   "margin": -8.65765412648519,
   "entropy": 1.0850799589136688,
   "spearman": 0.8241758241758241,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 13,
   "rank": 1,
   "margin": 1.6814672152201329,
   "entropy": 0.7386503115551671,
   "spearman": 0.8593406593406593,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 14,
   "rank": 10,
   "margin": -9.951443354288736,
   "entropy": 1.1718712836093303,
   "spearman": 0.6835164835164835,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 15,
   "rank": 10,
   "margin": -9.688493251800537,
   "entropy": 0.9353831052406607,
   "spearman": 0.7362637362637363,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 16,
   "rank": 10,
   "margin": -10.753314018249512,
   "entropy": 0.6825376731744933,
   "spearman": 0.7846153846153846,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 17,
   "rank": 14,
   "margin": -11.981592337290445,
   "entropy": 1.2278182017310826,
   "spearman": 0.6351648351648351,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 18,
   "rank": 10,
   "margin": -8.612460136413574,
   "entropy": 1.0508804744916376,
   "spearman": 0.7274725274725274,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 19,
   "rank": 13,
   "margin": -12.320711612701416,
   "entropy": 1.2316192242564425,
   "spearman": 0.6747252747252748,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 20,
   "rank": 14,
   "margin": -15.404692967732748,
   "entropy": 0.9613158702170245,
   "spearman": 0.5164835164835165,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 21,
   "rank": 10,
   "margin": -9.595308303833008,
   "entropy": 0.9650719613532736,
   "spearman": 0.7230769230769231,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 22,
   "rank": 9,
   "margin": -9.369369506835938,
   "entropy": 1.223323279234704,
   "spearman": 0.8021978021978022,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 23,
   "rank": 12,
   "margin": -13.68846845626831,
   "entropy": 0.7449818206859466,
   "spearman": 0.6703296703296703,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 24,
   "rank": 1,
   "margin": 0.5669790903727208,
   "entropy": 0.9700047817563224,
   "spearman": 0.8153846153846154,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 25,
   "rank": 10,
   "margin": -11.545512676239014,
   "entropy": 1.1227800657711866,
   "spearman": 0.7142857142857143,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 26,
   "rank": 13,
   "margin": -11.907694816589355,
   "entropy": 1.4078760412875533,
   "spearman": 0.6879120879120879,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 27,
   "rank": 9,
   "margin": -9.318997383117676,
   "entropy": 1.0193294749080442,
   "spearman": 0.7318681318681318,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 28,
   "rank": 9,
   "margin": -10.078191121419271,
   "entropy": 1.2026857453400779,
   "spearman": 0.7934065934065934,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 29,
   "rank": 11,
   "margin": -10.143146991729736,
   "entropy": 1.000202143072036,
   "spearman": 0.701098901098901,
   "correct": "mother",
   "pred": "sister"
  },
  {
   "traj": 30,
   "rank": 1,
   "margin": 1.3064327239990234,
   "entropy": 1.0589506608614887,
   "spearman": 0.832967032967033,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 31,
   "rank": 14,
   "margin": -12.720099449157715,
   "entropy": 1.2967586429540836,
   "spearman": 0.46373626373626375,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 32,
   "rank": 10,
   "margin": -7.851569493611654,
   "entropy": 1.2649950974611102,
   "spearman": 0.6923076923076923,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 33,
   "rank": 10,
   "margin": -10.575553894042969,
   "entropy": 1.157387330242886,
   "spearman": 0.7714285714285715,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 34,
   "rank": 2,
   "margin": -0.1960914929707851,
   "entropy": 1.265339563420026,
   "spearman": 0.8153846153846154,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 35,
   "rank": 10,
   "margin": -10.753314018249512,
   "entropy": 0.6825376731744933,
   "spearman": 0.7846153846153846,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 36,
   "rank": 9,
   "margin": -10.07837438583374,
   "entropy": 1.2022321980693287,
   "spearman": 0.7758241758241758,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 37,
   "rank": 11,
   "margin": -12.03713083267212,
   "entropy": 0.9416318410081586,
   "spearman": 0.8153846153846154,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 38,
   "rank": 14,
   "margin": -15.940930843353271,
   "entropy": 1.0423105588303767,
   "spearman": 0.7098901098901099,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 39,
   "rank": 10,
   "margin": -10.690700848897299,
   "entropy": 0.9106292292100996,
   "spearman": 0.7494505494505495,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 40,
   "rank": 13,
   "margin": -12.973217487335205,
   "entropy": 1.0200130751408976,
   "spearman": 0.6307692307692307,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 41,
   "rank": 11,
   "margin": -10.143683910369873,
   "entropy": 1.1764338124949456,
   "spearman": 0.6791208791208792,
   "correct": "father",
   "pred": "sister"
  },
  {
   "traj": 42,
   "rank": 11,
   "margin": -10.587874094645182,
   "entropy": 1.1611614984754268,
   "spearman": 0.7274725274725274,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 43,
   "rank": 13,
   "margin": -11.317754745483398,
   "entropy": 1.2107638836720702,
   "spearman": 0.6131868131868132,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 44,
   "rank": 14,
   "margin": -11.844386577606201,
   "entropy": 1.0321712199886812,
   "spearman": 0.643956043956044,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 45,
   "rank": 11,
   "margin": -10.143146991729736,
   "entropy": 1.000202143072036,
   "spearman": 0.701098901098901,
   "correct": "mother",
   "pred": "sister"
  },
  {
   "traj": 46,
   "rank": 13,
   "margin": -12.973217487335205,
   "entropy": 1.0200130751408976,
   "spearman": 0.6307692307692307,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 47,
   "rank": 12,
   "margin": -11.174745082855225,
   "entropy": 1.1297263960422765,
   "spearman": 0.7274725274725274,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 48,
   "rank": 12,
   "margin": -14.121274789174398,
   "entropy": 0.8673485899000863,
   "spearman": 0.7714285714285715,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 49,
   "rank": 10,
   "margin": -10.008797645568848,
   "entropy": 1.0691911136156202,
   "spearman": 0.6835164835164835,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 50,
   "rank": 13,
   "margin": -13.980965773264566,
   "entropy": 1.0048523632925137,
   "spearman": 0.6791208791208792,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 51,
   "rank": 13,
   "margin": -13.16803503036499,
   "entropy": 1.2082344924091206,
   "spearman": 0.6791208791208792,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 52,
   "rank": 13,
   "margin": -12.997047742207846,
   "entropy": 1.4279806781477622,
   "spearman": 0.6527472527472528,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 53,
   "rank": 14,
   "margin": -14.190618197123209,
   "entropy": 0.9652590176194809,
   "spearman": 0.42857142857142855,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 54,
   "rank": 10,
   "margin": -9.632436116536459,
   "entropy": 1.2978115253837947,
   "spearman": 0.7450549450549451,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 55,
   "rank": 14,
   "margin": -14.673632621765137,
   "entropy": 1.3660323182413594,
   "spearman": 0.4197802197802198,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 56,
   "rank": 13,
   "margin": -13.717884063720703,
   "entropy": 0.6385555696729772,
   "spearman": 0.5648351648351648,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 57,
   "rank": 14,
   "margin": -13.904067675272623,
   "entropy": 1.0675064483195185,
   "spearman": 0.42417582417582417,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 58,
   "rank": 1,
   "margin": 0.5637594858805333,
   "entropy": 1.1734086504790606,
   "spearman": 0.832967032967033,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 59,
   "rank": 11,
   "margin": -10.406675338745117,
   "entropy": 0.9996947778023287,
   "spearman": 0.7406593406593407,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 60,
   "rank": 2,
   "margin": -0.26320234934488873,
   "entropy": 1.0719733168865457,
   "spearman": 0.8153846153846154,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 61,
   "rank": 14,
   "margin": -15.940930843353271,
   "entropy": 1.0423105588303767,
   "spearman": 0.7098901098901099,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 62,
   "rank": 12,
   "margin": -12.12423324584961,
   "entropy": 1.2075919654270695,
   "spearman": 0.7582417582417582,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 63,
   "rank": 1,
   "margin": 0.46892531712849994,
   "entropy": 1.1829173820957628,
   "spearman": 0.8593406593406593,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 64,
   "rank": 14,
   "margin": -14.63234543800354,
   "entropy": 0.8815170401275148,
   "spearman": 0.48131868131868133,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 65,
   "rank": 13,
   "margin": -12.973217487335205,
   "entropy": 1.0200130751408976,
   "spearman": 0.6307692307692307,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 66,
   "rank": 11,
   "margin": -10.278847217559814,
   "entropy": 0.982480506587503,
   "spearman": 0.6967032967032967,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 67,
   "rank": 13,
   "margin": -13.31794802347819,
   "entropy": 0.9945991846924112,
   "spearman": 0.5296703296703297,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 68,
   "rank": 9,
   "margin": -8.421314239501953,
   "entropy": 1.2542815797285758,
   "spearman": 0.8417582417582418,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 69,
   "rank": 13,
   "margin": -12.53556219736735,
   "entropy": 1.067518474831872,
   "spearman": 0.7142857142857143,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 70,
   "rank": 11,
   "margin": -12.31851053237915,
   "entropy": 1.098352722296502,
   "spearman": 0.7142857142857143,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 71,
   "rank": 14,
   "margin": -11.604203542073568,
   "entropy": 1.1967543733295616,
   "spearman": 0.5692307692307692,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 72,
   "rank": 1,
   "margin": 1.1779235204060878,
   "entropy": 0.8356449894326836,
   "spearman": 0.8285714285714286,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 73,
   "rank": 11,
   "margin": -11.189310709635416,
   "entropy": 0.9210124322735187,
   "spearman": 0.7142857142857143,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 74,
   "rank": 11,
   "margin": -11.550645192464193,
   "entropy": 0.9929207490436988,
   "spearman": 0.7494505494505495,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 75,
   "rank": 13,
   "margin": -13.276015758514404,
   "entropy": 1.0922295019975088,
   "spearman": 0.6923076923076923,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 76,
   "rank": 13,
   "margin": -12.874957402547201,
   "entropy": 0.9847698604961753,
   "spearman": 0.6659340659340659,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 77,
   "rank": 14,
   "margin": -13.100122928619385,
   "entropy": 0.9968349900944964,
   "spearman": 0.5604395604395604,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 78,
   "rank": 1,
   "margin": 0.8750379085540771,
   "entropy": 0.918126131511887,
   "spearman": 0.8461538461538461,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 79,
   "rank": 11,
   "margin": -11.996849060058594,
   "entropy": 0.9921792356481618,
   "spearman": 0.6747252747252748,
   "correct": "father",
   "pred": "sister"
  },
  {
   "traj": 80,
   "rank": 2,
   "margin": -0.4473115603129063,
   "entropy": 1.233642349460434,
   "spearman": 0.8153846153846154,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 81,
   "rank": 11,
   "margin": -10.925927639007568,
   "entropy": 0.9528009789662903,
   "spearman": 0.6879120879120879,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 82,
   "rank": 14,
   "margin": -11.703263441721598,
   "entropy": 1.0638056560924791,
   "spearman": 0.5956043956043956,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 83,
   "rank": 9,
   "margin": -9.845712184906006,
   "entropy": 1.1038054391128862,
   "spearman": 0.7582417582417582,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 84,
   "rank": 2,
   "margin": -0.48796955744425397,
   "entropy": 1.1070771257440664,
   "spearman": 0.8373626373626374,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 85,
   "rank": 9,
   "margin": -9.341838518778482,
   "entropy": 1.0387659045890694,
   "spearman": 0.789010989010989,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 86,
   "rank": 9,
   "margin": -10.8203493754069,
   "entropy": 1.2085225026410729,
   "spearman": 0.8725274725274725,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 87,
   "rank": 13,
   "margin": -11.83907175064087,
   "entropy": 1.247496369406395,
   "spearman": 0.6791208791208792,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 88,
   "rank": 12,
   "margin": -10.555283546447754,
   "entropy": 1.2745233215356573,
   "spearman": 0.6923076923076923,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 89,
   "rank": 11,
   "margin": -11.853757699330648,
   "entropy": 1.0674500685366137,
   "spearman": 0.7054945054945055,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 90,
   "rank": 12,
   "margin": -13.027981122334799,
   "entropy": 0.8992880737796916,
   "spearman": 0.643956043956044,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 91,
   "rank": 13,
   "margin": -11.581449826558432,
   "entropy": 1.1635224380901645,
   "spearman": 0.5340659340659341,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 92,
   "rank": 10,
   "margin": -9.688493251800537,
   "entropy": 0.9353831052406607,
   "spearman": 0.7362637362637363,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 93,
   "rank": 14,
   "margin": -13.872509002685547,
   "entropy": 1.1471844274349023,
   "spearman": 0.38461538461538464,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 94,
   "rank": 11,
   "margin": -11.265257517496744,
   "entropy": 0.8237388271689076,
   "spearman": 0.832967032967033,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 95,
   "rank": 13,
   "margin": -14.142879009246826,
   "entropy": 0.9714412710642603,
   "spearman": 0.643956043956044,
   "correct": "mother",
   "pred": "sister"
  },
  {
   "traj": 96,
   "rank": 1,
   "margin": 0.18301677703857422,
   "entropy": 1.188294346763498,
   "spearman": 0.8153846153846154,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 97,
   "rank": 9,
   "margin": -8.689687410990397,
   "entropy": 1.137526564054769,
   "spearman": 0.8725274725274725,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 98,
   "rank": 14,
   "margin": -15.410875161488852,
   "entropy": 1.0934357192352406,
   "spearman": 0.7582417582417582,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 99,
   "rank": 13,
   "margin": -11.856807708740234,
   "entropy": 1.3737616403178274,
   "spearman": 0.6967032967032967,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 100,
   "rank": 13,
   "margin": -12.454861640930176,
   "entropy": 1.2865832866706426,
   "spearman": 0.7230769230769231,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 101,
   "rank": 13,
   "margin": -11.374983151753744,
   "entropy": 1.2691438460761577,
   "spearman": 0.5252747252747253,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 102,
   "rank": 10,
   "margin": -11.289779822031656,
   "entropy": 0.9892062013845249,
   "spearman": 0.7758241758241758,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 103,
   "rank": 12,
   "margin": -11.79966926574707,
   "entropy": 1.1791895574157618,
   "spearman": 0.7054945054945055,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 104,
   "rank": 9,
   "margin": -9.584932009379068,
   "entropy": 1.0781981993541914,
   "spearman": 0.7714285714285715,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 105,
   "rank": 11,
   "margin": -10.981908162434896,
   "entropy": 0.9043166154142511,
   "spearman": 0.7142857142857143,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 106,
   "rank": 14,
   "margin": -14.699254035949707,
   "entropy": 1.199944421528752,
   "spearman": 0.621978021978022,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 107,
   "rank": 10,
   "margin": -10.098409175872803,
   "entropy": 1.23526359944916,
   "spearman": 0.7186813186813187,
   "correct": "father",
   "pred": "sister"
  },
  {
   "traj": 108,
   "rank": 1,
   "margin": 0.09881385167439838,
   "entropy": 1.0085674641811444,
   "spearman": 0.8241758241758241,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 109,
   "rank": 9,
   "margin": -10.04230260848999,
   "entropy": 1.0573689575134095,
   "spearman": 0.8065934065934066,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 110,
   "rank": 10,
   "margin": -10.690700848897299,
   "entropy": 0.9106292292100996,
   "spearman": 0.7494505494505495,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 111,
   "rank": 11,
   "margin": -12.653223514556885,
   "entropy": 1.3961896394923856,
   "spearman": 0.7318681318681318,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 112,
   "rank": 2,
   "margin": -0.6091136932373047,
   "entropy": 1.1068208211481818,
   "spearman": 0.8021978021978022,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 113,
   "rank": 10,
   "margin": -9.595308303833008,
   "entropy": 0.9650719613532736,
   "spearman": 0.7230769230769231,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 114,
   "rank": 1,
   "margin": 0.8750379085540771,
   "entropy": 0.918126131511887,
   "spearman": 0.8461538461538461,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 115,
   "rank": 9,
   "margin": -9.649170557657877,
   "entropy": 0.9299519411542445,
   "spearman": 0.789010989010989,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 116,
   "rank": 9,
   "margin": -10.432926813761393,
   "entropy": 1.197814970430805,
   "spearman": 0.6703296703296703,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 117,
   "rank": 14,
   "margin": -15.963413715362549,
   "entropy": 1.0287361318844388,
   "spearman": 0.45054945054945056,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 118,
   "rank": 9,
   "margin": -9.11315123240153,
   "entropy": 1.175698832659951,
   "spearman": 0.7934065934065934,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 119,
   "rank": 14,
   "margin": -15.72808027267456,
   "entropy": 0.8302094893078702,
   "spearman": 0.42857142857142855,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 120,
   "rank": 14,
   "margin": -13.100122928619385,
   "entropy": 0.9968349900944964,
   "spearman": 0.5604395604395604,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 121,
   "rank": 14,
   "margin": -12.769133567810059,
   "entropy": 1.0316257775121962,
   "spearman": 0.4901098901098901,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 122,
   "rank": 13,
   "margin": -12.287638028462728,
   "entropy": 1.4172967199075313,
   "spearman": 0.7626373626373626,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 123,
   "rank": 13,
   "margin": -11.751428127288818,
   "entropy": 1.5082104028361816,
   "spearman": 0.7318681318681318,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 124,
   "rank": 9,
   "margin": -9.408243497212728,
   "entropy": 1.1273348640960892,
   "spearman": 0.7802197802197802,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 125,
   "rank": 10,
   "margin": -9.323992093404135,
   "entropy": 1.0527180061002004,
   "spearman": 0.7186813186813187,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 126,
   "rank": 12,
   "margin": -13.385574658711752,
   "entropy": 1.0967460877961552,
   "spearman": 0.6791208791208792,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 127,
   "rank": 14,
   "margin": -13.904067675272623,
   "entropy": 1.0675064483195185,
   "spearman": 0.42417582417582417,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 128,
   "rank": 13,
   "margin": -12.516871770222982,
   "entropy": 1.1163870642663045,
   "spearman": 0.6703296703296703,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 129,
   "rank": 9,
   "margin": -8.871369043986002,
   "entropy": 1.2156514567979362,
   "spearman": 0.7626373626373626,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 130,
   "rank": 14,
   "margin": -14.0691819190979,
   "entropy": 0.9024696639490403,
   "spearman": 0.46813186813186813,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 131,
   "rank": 9,
   "margin": -10.756242434183756,
   "entropy": 1.007086382202478,
   "spearman": 0.6967032967032967,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 132,
   "rank": 10,
   "margin": -9.521306355794271,
   "entropy": 1.0763517697510048,
   "spearman": 0.6395604395604395,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 133,
   "rank": 13,
   "margin": -11.342923164367676,
   "entropy": 1.258545756219624,
   "spearman": 0.7450549450549451,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 134,
   "rank": 1,
   "margin": 0.5669790903727208,
   "entropy": 0.9700047817563224,
   "spearman": 0.8153846153846154,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 135,
   "rank": 13,
   "margin": -10.059260686238607,
   "entropy": 1.0549563987562272,
   "spearman": 0.5428571428571428,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 136,
   "rank": 1,
   "margin": 0.16734997431437115,
   "entropy": 1.0861595116098257,
   "spearman": 0.8153846153846154,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 137,
   "rank": 2,
   "margin": -0.9767773946126308,
   "entropy": 1.0309363475684408,
   "spearman": 0.7846153846153846,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 138,
   "rank": 1,
   "margin": 0.31621281305948834,
   "entropy": 1.2778883030856256,
   "spearman": 0.7714285714285715,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 139,
   "rank": 1,
   "margin": 0.3544179598490391,
   "entropy": 1.4736786508268693,
   "spearman": 0.8285714285714286,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 140,
   "rank": 9,
   "margin": -11.170923233032227,
   "entropy": 1.0950601311294959,
   "spearman": 0.7362637362637363,
   "correct": "father",
   "pred": "sister"
  },
  {
   "traj": 141,
   "rank": 2,
   "margin": -0.5175665219624843,
   "entropy": 1.0986481652192825,
   "spearman": 0.8505494505494505,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 142,
   "rank": 9,
   "margin": -8.888019243876139,
   "entropy": 0.7872578122364369,
   "spearman": 0.6791208791208792,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 143,
   "rank": 9,
   "margin": -10.318914731343588,
   "entropy": 0.8952962015510537,
   "spearman": 0.6879120879120879,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 144,
   "rank": 9,
   "margin": -10.065321922302246,
   "entropy": 1.1479709778465048,
   "spearman": 0.8065934065934066,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 145,
   "rank": 11,
   "margin": -9.501461664835611,
   "entropy": 1.115054547912322,
   "spearman": 0.7186813186813187,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 146,
   "rank": 12,
   "margin": -11.195123990376791,
   "entropy": 1.0934723409459128,
   "spearman": 0.5824175824175825,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 147,
   "rank": 11,
   "margin": -10.063740730285645,
   "entropy": 1.185727143103449,
   "spearman": 0.6615384615384615,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 148,
   "rank": 13,
   "margin": -11.83907175064087,
   "entropy": 1.247496369406395,
   "spearman": 0.6791208791208792,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 149,
   "rank": 14,
   "margin": -16.778464317321777,
   "entropy": 0.9240117554105601,
   "spearman": 0.6043956043956044,
   "correct": "daughter",
   "pred": "nephew"
  }
 ],
 "norm600": [
  {
   "traj": 0,
   "rank": 12,
   "margin": -4.974640369415283,
   "entropy": 1.9407461488617024,
   "spearman": 0.7230769230769231,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 1,
   "rank": 9,
   "margin": -5.794660568237305,
   "entropy": 1.7999397042664336,
   "spearman": 0.9164835164835164,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 2,
   "rank": 9,
   "margin": -3.4117441177368164,
   "entropy": 2.0263125037050482,
   "spearman": 0.8549450549450549,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 3,
   "rank": 9,
   "margin": -4.476745923360189,
   "entropy": 1.8164337056052338,
   "spearman": 0.8637362637362638,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 4,
   "rank": 9,
   "margin": -4.377400875091553,
   "entropy": 1.8633112991398741,
   "spearman": 0.8197802197802198,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 5,
   "rank": 2,
   "margin": -0.048334360122680664,
   "entropy": 1.9555167948367347,
   "spearman": 0.8373626373626374,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 6,
   "rank": 11,
   "margin": -4.885773181915283,
   "entropy": 1.9025624987706227,
   "spearman": 0.810989010989011,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 7,
   "rank": 10,
   "margin": -5.718883991241455,
   "entropy": 1.8268893149408585,
   "spearman": 0.810989010989011,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 8,
   "rank": 10,
   "margin": -5.0809245109558105,
   "entropy": 1.9516708208321332,
   "spearman": 0.8241758241758241,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 9,
   "rank": 9,
   "margin": -4.064234495162964,
   "entropy": 1.9934184426894732,
   "spearman": 0.832967032967033,
   "correct": "mother",
   "pred": "grandmother"
  },
  {
   "traj": 10,
   "rank": 11,
   "margin": -5.147841612497965,
   "entropy": 1.6230923140419016,
   "spearman": 0.8417582417582418,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 11,
   "rank": 14,
   "margin": -6.375750541687012,
   "entropy": 1.938530085639747,
   "spearman": 0.42857142857142855,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 12,
   "rank": 12,
   "margin": -5.551577568054199,
   "entropy": 1.7020709452107483,
   "spearman": 0.8285714285714286,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 13,
   "rank": 4,
   "margin": -0.6635231971740723,
   "entropy": 1.9411845922101163,
   "spearman": 0.7978021978021979,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 14,
   "rank": 10,
   "margin": -4.6564600467681885,
   "entropy": 1.864207225119432,
   "spearman": 0.8461538461538461,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 15,
   "rank": 10,
   "margin": -5.265288829803467,
   "entropy": 1.9688138087200118,
   "spearman": 0.8197802197802198,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 16,
   "rank": 10,
   "margin": -4.882763226826986,
   "entropy": 1.8611783148384997,
   "spearman": 0.9208791208791208,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 17,
   "rank": 13,
   "margin": -7.141720771789551,
   "entropy": 1.8872342982575065,
   "spearman": 0.6571428571428571,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 18,
   "rank": 10,
   "margin": -4.869179725646973,
   "entropy": 1.8098586329444006,
   "spearman": 0.8153846153846154,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 19,
   "rank": 11,
   "margin": -6.104192733764648,
   "entropy": 1.6589596870939995,
   "spearman": 0.7802197802197802,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 20,
   "rank": 12,
   "margin": -6.068727970123291,
   "entropy": 1.9190746284123108,
   "spearman": 0.6175824175824176,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 21,
   "rank": 9,
   "margin": -4.597443580627441,
   "entropy": 1.7913405054689684,
   "spearman": 0.8461538461538461,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 22,
   "rank": 12,
   "margin": -6.401092290878296,
   "entropy": 1.652404051936987,
   "spearman": 0.7582417582417582,
   "correct": "son",
   "pred": "sister"
  },
  {
   "traj": 23,
   "rank": 9,
   "margin": -4.822499593098959,
   "entropy": 1.873997218042477,
   "spearman": 0.8725274725274725,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 24,
   "rank": 3,
   "margin": -0.3293396234512329,
   "entropy": 1.8764639441491457,
   "spearman": 0.7582417582417582,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 25,
   "rank": 9,
   "margin": -5.459131558736166,
   "entropy": 1.7521584380694502,
   "spearman": 0.8725274725274725,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 26,
   "rank": 11,
   "margin": -6.7259931564331055,
   "entropy": 1.6502759112090444,
   "spearman": 0.832967032967033,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 27,
   "rank": 11,
   "margin": -5.494469881057739,
   "entropy": 1.9492588311848063,
   "spearman": 0.8021978021978022,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 28,
   "rank": 10,
   "margin": -5.718883991241455,
   "entropy": 1.8268893149408585,
   "spearman": 0.810989010989011,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 29,
   "rank": 9,
   "margin": -4.523700952529907,
   "entropy": 1.9270562642062385,
   "spearman": 0.8813186813186813,
   "correct": "mother",
   "pred": "grandmother"
  },
  {
   "traj": 30,
   "rank": 4,
   "margin": -0.493755578994751,
   "entropy": 1.9298130299334548,
   "spearman": 0.7758241758241758,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 31,
   "rank": 14,
   "margin": -6.530060291290283,
   "entropy": 1.9797565917155615,
   "spearman": 0.46813186813186813,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 32,
   "rank": 10,
   "margin": -5.4509968757629395,
   "entropy": 1.7483520713346694,
   "spearman": 0.789010989010989,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 33,
   "rank": 11,
   "margin": -5.254821062088013,
   "entropy": 1.9868889144255646,
   "spearman": 0.8241758241758241,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 34,
   "rank": 4,
   "margin": -0.4663982391357422,
   "entropy": 2.020150516208125,
   "spearman": 0.7846153846153846,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 35,
   "rank": 10,
   "margin": -4.882763226826986,
   "entropy": 1.8611783148384997,
   "spearman": 0.9208791208791208,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 36,
   "rank": 12,
   "margin": -7.168400526046753,
   "entropy": 1.6765698413259107,
   "spearman": 0.7846153846153846,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 37,
   "rank": 12,
   "margin": -6.357509613037109,
   "entropy": 1.714470152122647,
   "spearman": 0.789010989010989,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 38,
   "rank": 12,
   "margin": -5.613502025604248,
   "entropy": 1.9739332551180224,
   "spearman": 0.6703296703296703,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 39,
   "rank": 11,
   "margin": -5.069281578063965,
   "entropy": 1.575441199582991,
   "spearman": 0.8681318681318682,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 40,
   "rank": 9,
   "margin": -4.98655923207601,
   "entropy": 1.8303682817905533,
   "spearman": 0.8065934065934066,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 41,
   "rank": 10,
   "margin": -4.92244815826416,
   "entropy": 1.591474193713623,
   "spearman": 0.8285714285714286,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 42,
   "rank": 13,
   "margin": -6.214365243911743,
   "entropy": 1.7449941229629697,
   "spearman": 0.7186813186813187,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 43,
   "rank": 12,
   "margin": -5.514322757720947,
   "entropy": 1.6222938991543427,
   "spearman": 0.7978021978021979,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 44,
   "rank": 12,
   "margin": -6.260699590047201,
   "entropy": 1.7499593547149055,
   "spearman": 0.7318681318681318,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 45,
   "rank": 9,
   "margin": -4.523700952529907,
   "entropy": 1.9270562642062385,
   "spearman": 0.8813186813186813,
   "correct": "mother",
   "pred": "grandmother"
  },
  {
   "traj": 46,
   "rank": 9,
   "margin": -4.98655923207601,
   "entropy": 1.8303682817905533,
   "spearman": 0.8065934065934066,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 47,
   "rank": 11,
   "margin": -5.0269389152526855,
   "entropy": 1.902436093547831,
   "spearman": 0.832967032967033,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 48,
   "rank": 12,
   "margin": -5.935351610183716,
   "entropy": 1.7703056565971478,
   "spearman": 0.7450549450549451,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 49,
   "rank": 10,
   "margin": -5.531249284744263,
   "entropy": 1.783852004885986,
   "spearman": 0.789010989010989,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 50,
   "rank": 9,
   "margin": -4.721928914388021,
   "entropy": 1.932537541581102,
   "spearman": 0.8637362637362638,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 51,
   "rank": 12,
   "margin": -4.916834354400635,
   "entropy": 1.9485525253808849,
   "spearman": 0.6791208791208792,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 52,
   "rank": 13,
   "margin": -7.036256790161133,
   "entropy": 1.653112392694039,
   "spearman": 0.643956043956044,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 53,
   "rank": 14,
   "margin": -6.375750541687012,
   "entropy": 1.938530085639747,
   "spearman": 0.42857142857142855,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 54,
   "rank": 12,
   "margin": -6.125791072845459,
   "entropy": 1.6459777196182566,
   "spearman": 0.7406593406593407,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 55,
   "rank": 12,
   "margin": -6.2532172203063965,
   "entropy": 1.928879038265464,
   "spearman": 0.5560439560439561,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 56,
   "rank": 9,
   "margin": -4.564066727956137,
   "entropy": 1.869326063111598,
   "spearman": 0.8417582417582418,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 57,
   "rank": 11,
   "margin": -6.193284352620443,
   "entropy": 1.8463316688599503,
   "spearman": 0.5868131868131868,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 58,
   "rank": 1,
   "margin": 0.741157054901123,
   "entropy": 1.8843443075368065,
   "spearman": 0.8065934065934066,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 59,
   "rank": 10,
   "margin": -5.666920820871988,
   "entropy": 1.523098532655886,
   "spearman": 0.9208791208791208,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 60,
   "rank": 4,
   "margin": -0.4155604839324951,
   "entropy": 1.9536250559409343,
   "spearman": 0.7406593406593407,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 61,
   "rank": 12,
   "margin": -5.613502025604248,
   "entropy": 1.9739332551180224,
   "spearman": 0.6703296703296703,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 62,
   "rank": 13,
   "margin": -7.400002956390381,
   "entropy": 1.6559780597301377,
   "spearman": 0.6615384615384615,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 63,
   "rank": 1,
   "margin": 0.8043293952941895,
   "entropy": 1.830719379430151,
   "spearman": 0.8065934065934066,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 64,
   "rank": 13,
   "margin": -6.392887830734253,
   "entropy": 1.9805534821756576,
   "spearman": 0.5560439560439561,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 65,
   "rank": 9,
   "margin": -4.98655923207601,
   "entropy": 1.8303682817905533,
   "spearman": 0.8065934065934066,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 66,
   "rank": 12,
   "margin": -6.141310214996338,
   "entropy": 1.6993697286089773,
   "spearman": 0.7978021978021979,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 67,
   "rank": 9,
   "margin": -4.358644644419352,
   "entropy": 1.897912132318481,
   "spearman": 0.7230769230769231,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 68,
   "rank": 14,
   "margin": -6.446261644363403,
   "entropy": 1.8516686783534426,
   "spearman": 0.7978021978021979,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 69,
   "rank": 12,
   "margin": -6.288677215576172,
   "entropy": 1.763113950378168,
   "spearman": 0.7802197802197802,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 70,
   "rank": 12,
   "margin": -5.258160829544067,
   "entropy": 1.9963784574682535,
   "spearman": 0.6175824175824176,
   "correct": "brother",
   "pred": "grandson"
  },
  {
   "traj": 71,
   "rank": 13,
   "margin": -5.693238139152527,
   "entropy": 1.9776737735290748,
   "spearman": 0.6791208791208792,
   "correct": "daughter",
   "pred": "grandmother"
  },
  {
   "traj": 72,
   "rank": 3,
   "margin": -0.9364315668741856,
   "entropy": 1.775958798474439,
   "spearman": 0.7494505494505495,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 73,
   "rank": 9,
   "margin": -4.06852134068807,
   "entropy": 1.9062119599657086,
   "spearman": 0.8681318681318682,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 74,
   "rank": 9,
   "margin": -4.387490590413412,
   "entropy": 1.8376579102246222,
   "spearman": 0.9120879120879121,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 75,
   "rank": 13,
   "margin": -7.149836540222168,
   "entropy": 1.7518220058798757,
   "spearman": 0.6703296703296703,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 76,
   "rank": 11,
   "margin": -6.603999773661295,
   "entropy": 1.6929323258090985,
   "spearman": 0.7934065934065934,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 77,
   "rank": 11,
   "margin": -5.727621793746948,
   "entropy": 1.8824012104062677,
   "spearman": 0.6571428571428571,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 78,
   "rank": 1,
   "margin": 0.053218722343444824,
   "entropy": 1.8325661361367123,
   "spearman": 0.8593406593406593,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 79,
   "rank": 10,
   "margin": -4.736398061116537,
   "entropy": 1.900577492665554,
   "spearman": 0.8417582417582418,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 80,
   "rank": 5,
   "margin": -0.5107431411743164,
   "entropy": 2.011259228719196,
   "spearman": 0.7362637362637363,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 81,
   "rank": 9,
   "margin": -4.459468841552734,
   "entropy": 1.7243056110490136,
   "spearman": 0.8857142857142857,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 82,
   "rank": 11,
   "margin": -6.171662092208862,
   "entropy": 1.7390057682776023,
   "spearman": 0.6791208791208792,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 83,
   "rank": 12,
   "margin": -7.048068284988403,
   "entropy": 1.6698594987734652,
   "spearman": 0.7846153846153846,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 84,
   "rank": 1,
   "margin": 0.6118929386138916,
   "entropy": 1.8553034643672186,
   "spearman": 0.8065934065934066,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 85,
   "rank": 11,
   "margin": -4.709673086802164,
   "entropy": 1.9246806757336903,
   "spearman": 0.9076923076923077,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 86,
   "rank": 12,
   "margin": -6.83446741104126,
   "entropy": 1.7673848593972294,
   "spearman": 0.789010989010989,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 87,
   "rank": 12,
   "margin": -5.723369359970093,
   "entropy": 1.92194179348874,
   "spearman": 0.6263736263736264,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 88,
   "rank": 9,
   "margin": -4.229699432849884,
   "entropy": 1.962895252700565,
   "spearman": 0.9428571428571428,
   "correct": "mother",
   "pred": "grandmother"
  },
  {
   "traj": 89,
   "rank": 9,
   "margin": -4.261561393737793,
   "entropy": 1.9680647040215444,
   "spearman": 0.8945054945054945,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 90,
   "rank": 9,
   "margin": -4.780868689219156,
   "entropy": 1.8126472601291483,
   "spearman": 0.8153846153846154,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 91,
   "rank": 9,
   "margin": -4.359487295150757,
   "entropy": 1.9517904703537572,
   "spearman": 0.7142857142857143,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 92,
   "rank": 10,
   "margin": -5.265288829803467,
   "entropy": 1.9688138087200118,
   "spearman": 0.8197802197802198,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 93,
   "rank": 13,
   "margin": -6.0954084396362305,
   "entropy": 1.9393969635948318,
   "spearman": 0.5208791208791209,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 94,
   "rank": 11,
   "margin": -6.025960286458334,
   "entropy": 1.7739177989043646,
   "spearman": 0.7758241758241758,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 95,
   "rank": 9,
   "margin": -3.9639040629069004,
   "entropy": 1.9482610623116345,
   "spearman": 0.8989010989010989,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 96,
   "rank": 1,
   "margin": 0.063323974609375,
   "entropy": 1.94276617172793,
   "spearman": 0.789010989010989,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 97,
   "rank": 11,
   "margin": -5.738601366678873,
   "entropy": 1.811805711747155,
   "spearman": 0.9076923076923077,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 98,
   "rank": 14,
   "margin": -7.3152882258097325,
   "entropy": 1.5471248983744548,
   "spearman": 0.7362637362637363,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 99,
   "rank": 11,
   "margin": -5.765298843383789,
   "entropy": 1.8384958631215536,
   "spearman": 0.7538461538461538,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 100,
   "rank": 12,
   "margin": -6.298738479614258,
   "entropy": 1.747362087424173,
   "spearman": 0.6835164835164835,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 101,
   "rank": 9,
   "margin": -4.191771030426025,
   "entropy": 1.8744346713879891,
   "spearman": 0.8461538461538461,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 102,
   "rank": 9,
   "margin": -5.208722114562988,
   "entropy": 1.7089344463053753,
   "spearman": 0.9164835164835164,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 103,
   "rank": 9,
   "margin": -4.098689715067545,
   "entropy": 1.941397960486155,
   "spearman": 0.8593406593406593,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 104,
   "rank": 12,
   "margin": -5.96269416809082,
   "entropy": 1.9670972894955399,
   "spearman": 0.7494505494505495,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 105,
   "rank": 9,
   "margin": -4.1952805519104,
   "entropy": 1.9400805704590025,
   "spearman": 0.8505494505494505,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 106,
   "rank": 11,
   "margin": -5.893885135650635,
   "entropy": 1.937529764098971,
   "spearman": 0.6791208791208792,
   "correct": "daughter",
   "pred": "sister"
  },
  {
   "traj": 107,
   "rank": 9,
   "margin": -4.5050270557403564,
   "entropy": 1.7801354984145106,
   "spearman": 0.8065934065934066,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 108,
   "rank": 4,
   "margin": -0.7313791910807286,
   "entropy": 1.8968196589326762,
   "spearman": 0.7846153846153846,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 109,
   "rank": 12,
   "margin": -6.934737205505371,
   "entropy": 1.7546184542715235,
   "spearman": 0.8285714285714286,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 110,
   "rank": 11,
   "margin": -5.069281578063965,
   "entropy": 1.575441199582991,
   "spearman": 0.8681318681318682,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 111,
   "rank": 11,
   "margin": -4.898724794387817,
   "entropy": 1.9708109463939247,
   "spearman": 0.701098901098901,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 112,
   "rank": 3,
   "margin": -0.4154365857442226,
   "entropy": 1.734119000673055,
   "spearman": 0.8417582417582418,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 113,
   "rank": 9,
   "margin": -4.597443580627441,
   "entropy": 1.7913405054689684,
   "spearman": 0.8461538461538461,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 114,
   "rank": 1,
   "margin": 0.053218722343444824,
   "entropy": 1.8325661361367123,
   "spearman": 0.8593406593406593,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 115,
   "rank": 9,
   "margin": -5.577717622121176,
   "entropy": 1.7459554745246018,
   "spearman": 0.8769230769230769,
   "correct": "son",
   "pred": "nephew"
  },
  {
   "traj": 116,
   "rank": 9,
   "margin": -4.20704984664917,
   "entropy": 1.8267494374792839,
   "spearman": 0.8065934065934066,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 117,
   "rank": 12,
   "margin": -5.993052164713541,
   "entropy": 1.9264135170645271,
   "spearman": 0.5736263736263736,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 118,
   "rank": 10,
   "margin": -4.938788890838623,
   "entropy": 1.8523013321059416,
   "spearman": 0.9560439560439561,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 119,
   "rank": 12,
   "margin": -6.809528112411499,
   "entropy": 1.4684343789449275,
   "spearman": 0.5956043956043956,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 120,
   "rank": 11,
   "margin": -5.727621793746948,
   "entropy": 1.8824012104062677,
   "spearman": 0.6571428571428571,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 121,
   "rank": 12,
   "margin": -6.43485689163208,
   "entropy": 1.8485352107679331,
   "spearman": 0.5912087912087912,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 122,
   "rank": 13,
   "margin": -6.822906732559204,
   "entropy": 1.9099950319198646,
   "spearman": 0.701098901098901,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 123,
   "rank": 13,
   "margin": -6.480743885040283,
   "entropy": 1.9668161367998107,
   "spearman": 0.7098901098901099,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 124,
   "rank": 11,
   "margin": -5.07346248626709,
   "entropy": 1.8874324168897136,
   "spearman": 0.8593406593406593,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 125,
   "rank": 10,
   "margin": -5.38958477973938,
   "entropy": 1.961609303046803,
   "spearman": 0.8197802197802198,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 126,
   "rank": 12,
   "margin": -6.579655647277832,
   "entropy": 1.7145853049496456,
   "spearman": 0.7406593406593407,
   "correct": "brother",
   "pred": "nephew"
  },
  {
   "traj": 127,
   "rank": 11,
   "margin": -6.193284352620443,
   "entropy": 1.8463316688599503,
   "spearman": 0.5868131868131868,
   "correct": "daughter",
   "pred": "nephew"
  },
  {
   "traj": 128,
   "rank": 11,
   "margin": -6.0013227462768555,
   "entropy": 1.8033274396361443,
   "spearman": 0.7142857142857143,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 129,
   "rank": 9,
   "margin": -4.451745271682739,
   "entropy": 1.8713492155012765,
   "spearman": 0.9516483516483516,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 130,
   "rank": 12,
   "margin": -5.2386863231658936,
   "entropy": 2.0388665795938583,
   "spearman": 0.578021978021978,
   "correct": "daughter",
   "pred": "grandmother"
  },
  {
   "traj": 131,
   "rank": 10,
   "margin": -4.743149757385254,
   "entropy": 1.72970273634954,
   "spearman": 0.7714285714285715,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 132,
   "rank": 9,
   "margin": -5.621805429458618,
   "entropy": 1.5197066365905971,
   "spearman": 0.8285714285714286,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 133,
   "rank": 12,
   "margin": -6.631389141082764,
   "entropy": 1.4361561395590328,
   "spearman": 0.7758241758241758,
   "correct": "brother",
   "pred": "sister"
  },
  {
   "traj": 134,
   "rank": 3,
   "margin": -0.3293396234512329,
   "entropy": 1.8764639441491457,
   "spearman": 0.7582417582417582,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 135,
   "rank": 9,
   "margin": -4.567924976348877,
   "entropy": 1.8366252898192967,
   "spearman": 0.7054945054945055,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 136,
   "rank": 1,
   "margin": 0.6033284664154053,
   "entropy": 1.8775286558450526,
   "spearman": 0.7978021978021979,
   "correct": "sister",
   "pred": "sister"
  },
  {
   "traj": 137,
   "rank": 4,
   "margin": -0.5223538875579834,
   "entropy": 1.8721367996625915,
   "spearman": 0.7494505494505495,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 138,
   "rank": 6,
   "margin": -1.4498579502105713,
   "entropy": 1.892353912720592,
   "spearman": 0.7230769230769231,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 139,
   "rank": 2,
   "margin": -0.14193105697631836,
   "entropy": 1.9213594373350529,
   "spearman": 0.8153846153846154,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 140,
   "rank": 9,
   "margin": -5.134891986846924,
   "entropy": 1.5094934918799299,
   "spearman": 0.8901098901098901,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 141,
   "rank": 6,
   "margin": -1.7704960505167637,
   "entropy": 1.8268160080275855,
   "spearman": 0.7582417582417582,
   "correct": "sister",
   "pred": "nephew"
  },
  {
   "traj": 142,
   "rank": 9,
   "margin": -4.7049620151519775,
   "entropy": 1.8515496046082804,
   "spearman": 0.9032967032967033,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 143,
   "rank": 10,
   "margin": -5.641311168670654,
   "entropy": 1.8500507642072717,
   "spearman": 0.8021978021978022,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 144,
   "rank": 10,
   "margin": -5.0809245109558105,
   "entropy": 1.9516708208321332,
   "spearman": 0.8241758241758241,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 145,
   "rank": 9,
   "margin": -5.249622186024984,
   "entropy": 1.8305205423421618,
   "spearman": 0.8153846153846154,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 146,
   "rank": 9,
   "margin": -6.046395142873129,
   "entropy": 1.6821315096046459,
   "spearman": 0.7934065934065934,
   "correct": "mother",
   "pred": "nephew"
  },
  {
   "traj": 147,
   "rank": 9,
   "margin": -5.533648252487183,
   "entropy": 1.9070321499554208,
   "spearman": 0.8505494505494505,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 148,
   "rank": 12,
   "margin": -5.723369359970093,
   "entropy": 1.92194179348874,
   "spearman": 0.6263736263736264,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 149,
   "rank": 11,
   "margin": -5.753758907318115,
   "entropy": 1.9536043975753683,
   "spearman": 0.7098901098901099,
   "correct": "daughter",
   "pred": "grandmother"
  }
 ],
 "norm1800": [
  {
   "traj": 0,
   "rank": 13,
   "margin": -6.934591233730316,
   "entropy": 1.329245090285613,
   "spearman": 0.5868131868131868,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 1,
   "rank": 13,
   "margin": -5.927269220352173,
   "entropy": 1.6683641398682503,
   "spearman": 0.6,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 2,
   "rank": 10,
   "margin": -5.4645702838897705,
   "entropy": 1.6491530940647607,
   "spearman": 0.7450549450549451,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 3,
   "rank": 10,
   "margin": -5.27519166469574,
   "entropy": 1.4405386784749554,
   "spearman": 0.8021978021978022,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 4,
   "rank": 9,
   "margin": -4.202670574188232,
   "entropy": 1.82129873086764,
   "spearman": 0.7494505494505495,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 5,
   "rank": 6,
   "margin": -1.8699684143066406,
   "entropy": 1.6701035547724317,
   "spearman": 0.7142857142857143,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 6,
   "rank": 9,
   "margin": -4.326457977294922,
   "entropy": 1.8073438867691303,
   "spearman": 0.832967032967033,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 7,
   "rank": 9,
   "margin": -4.752919316291809,
   "entropy": 1.5496243300697499,
   "spearman": 0.8593406593406593,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 8,
   "rank": 9,
   "margin": -4.394303143024445,
   "entropy": 1.7460934512300792,
   "spearman": 0.8021978021978022,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 9,
   "rank": 10,
   "margin": -4.672064542770386,
   "entropy": 1.7021543849802816,
   "spearman": 0.7362637362637363,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 10,
   "rank": 7,
   "margin": -4.149152040481567,
   "entropy": 1.4310311104796618,
   "spearman": 0.9340659340659341,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 11,
   "rank": 12,
   "margin": -6.308126211166382,
   "entropy": 1.7158397284175522,
   "spearman": 0.5252747252747253,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 12,
   "rank": 8,
   "margin": -3.688905119895935,
   "entropy": 1.7213596677783822,
   "spearman": 0.8637362637362638,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 13,
   "rank": 7,
   "margin": -2.3559091091156006,
   "entropy": 1.823372519383655,
   "spearman": 0.6703296703296703,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 14,
   "rank": 12,
   "margin": -6.2122273445129395,
   "entropy": 1.5581243571244556,
   "spearman": 0.7802197802197802,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 15,
   "rank": 11,
   "margin": -5.6928651332855225,
   "entropy": 1.803778010566667,
   "spearman": 0.7626373626373626,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 16,
   "rank": 7,
   "margin": -3.1761115193367004,
   "entropy": 1.6654871101942514,
   "spearman": 0.810989010989011,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 17,
   "rank": 13,
   "margin": -7.222388505935669,
   "entropy": 1.6676747074767817,
   "spearman": 0.4989010989010989,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 18,
   "rank": 11,
   "margin": -5.671414375305176,
   "entropy": 1.7626123537726488,
   "spearman": 0.7582417582417582,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 19,
   "rank": 12,
   "margin": -6.340274810791016,
   "entropy": 1.5800458085290319,
   "spearman": 0.7538461538461538,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 20,
   "rank": 11,
   "margin": -6.202472805976868,
   "entropy": 1.6580567053769961,
   "spearman": 0.5032967032967033,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 21,
   "rank": 11,
   "margin": -5.0409557819366455,
   "entropy": 1.6129273955861352,
   "spearman": 0.7934065934065934,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 22,
   "rank": 9,
   "margin": -5.289370179176331,
   "entropy": 1.5902670906160254,
   "spearman": 0.8725274725274725,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 23,
   "rank": 10,
   "margin": -5.554054021835327,
   "entropy": 1.547853009402983,
   "spearman": 0.7978021978021979,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 24,
   "rank": 6,
   "margin": -1.2548691034317017,
   "entropy": 1.8437330678564632,
   "spearman": 0.6967032967032967,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 25,
   "rank": 9,
   "margin": -4.584206402301788,
   "entropy": 1.490706051054271,
   "spearman": 0.8461538461538461,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 26,
   "rank": 12,
   "margin": -5.454053163528442,
   "entropy": 1.56763088880909,
   "spearman": 0.6879120879120879,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 27,
   "rank": 9,
   "margin": -4.377641916275024,
   "entropy": 1.5231718884301386,
   "spearman": 0.810989010989011,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 28,
   "rank": 9,
   "margin": -4.752919316291809,
   "entropy": 1.5496243300697499,
   "spearman": 0.8593406593406593,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 29,
   "rank": 10,
   "margin": -5.632155537605286,
   "entropy": 1.5693376663012093,
   "spearman": 0.8065934065934066,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 30,
   "rank": 5,
   "margin": -1.6916333436965942,
   "entropy": 1.69822321219475,
   "spearman": 0.7186813186813187,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 31,
   "rank": 11,
   "margin": -6.033734679222107,
   "entropy": 1.672134447164938,
   "spearman": 0.578021978021978,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 32,
   "rank": 12,
   "margin": -5.90268349647522,
   "entropy": 1.6409270802722549,
   "spearman": 0.8065934065934066,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 33,
   "rank": 9,
   "margin": -3.906448245048523,
   "entropy": 1.7704833047337198,
   "spearman": 0.7978021978021979,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 34,
   "rank": 7,
   "margin": -2.2615081071853638,
   "entropy": 1.7823085249049402,
   "spearman": 0.6659340659340659,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 35,
   "rank": 7,
   "margin": -3.1761115193367004,
   "entropy": 1.6654871101942514,
   "spearman": 0.810989010989011,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 36,
   "rank": 9,
   "margin": -4.744590878486633,
   "entropy": 1.6020291692104558,
   "spearman": 0.9032967032967033,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 37,
   "rank": 12,
   "margin": -6.267730712890625,
   "entropy": 1.5358747656527747,
   "spearman": 0.7406593406593407,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 38,
   "rank": 12,
   "margin": -5.39274275302887,
   "entropy": 1.7360616143454504,
   "spearman": 0.7450549450549451,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 39,
   "rank": 8,
   "margin": -4.3127923011779785,
   "entropy": 1.240831895008948,
   "spearman": 0.8725274725274725,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 40,
   "rank": 9,
   "margin": -4.179586887359619,
   "entropy": 1.8288856658758939,
   "spearman": 0.8065934065934066,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 41,
   "rank": 11,
   "margin": -6.243795871734619,
   "entropy": 1.4873407007947181,
   "spearman": 0.8681318681318682,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 42,
   "rank": 9,
   "margin": -5.160699248313904,
   "entropy": 1.3692865451002465,
   "spearman": 0.8901098901098901,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 43,
   "rank": 9,
   "margin": -4.908979177474976,
   "entropy": 1.6039527852564992,
   "spearman": 0.8989010989010989,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 44,
   "rank": 11,
   "margin": -7.105417966842651,
   "entropy": 1.4347074250895524,
   "spearman": 0.7626373626373626,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 45,
   "rank": 10,
   "margin": -5.632155537605286,
   "entropy": 1.5693376663012093,
   "spearman": 0.8065934065934066,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 46,
   "rank": 9,
   "margin": -4.179586887359619,
   "entropy": 1.8288856658758939,
   "spearman": 0.8065934065934066,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 47,
   "rank": 8,
   "margin": -2.8952956199645996,
   "entropy": 1.8006859419753842,
   "spearman": 0.8241758241758241,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 48,
   "rank": 13,
   "margin": -6.429178476333618,
   "entropy": 1.5293675018832202,
   "spearman": 0.6571428571428571,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 49,
   "rank": 11,
   "margin": -5.992792367935181,
   "entropy": 1.5597435732651244,
   "spearman": 0.8021978021978022,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 50,
   "rank": 9,
   "margin": -4.05417788028717,
   "entropy": 1.8227164494891894,
   "spearman": 0.810989010989011,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 51,
   "rank": 13,
   "margin": -6.578564345836639,
   "entropy": 1.600796555902455,
   "spearman": 0.6307692307692307,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 52,
   "rank": 12,
   "margin": -6.912806272506714,
   "entropy": 1.6793993798435105,
   "spearman": 0.7186813186813187,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 53,
   "rank": 12,
   "margin": -6.308126211166382,
   "entropy": 1.7158397284175522,
   "spearman": 0.5252747252747253,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 54,
   "rank": 9,
   "margin": -4.414642810821533,
   "entropy": 1.546524329819963,
   "spearman": 0.8901098901098901,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 55,
   "rank": 11,
   "margin": -6.768702507019043,
   "entropy": 1.6654457564988097,
   "spearman": 0.6087912087912087,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 56,
   "rank": 9,
   "margin": -5.093491315841675,
   "entropy": 1.6148002009323899,
   "spearman": 0.7758241758241758,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 57,
   "rank": 11,
   "margin": -6.109084606170654,
   "entropy": 1.6653072128554882,
   "spearman": 0.6351648351648351,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 58,
   "rank": 6,
   "margin": -1.765323519706726,
   "entropy": 1.7022472995448708,
   "spearman": 0.6527472527472528,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 59,
   "rank": 7,
   "margin": -3.43354469537735,
   "entropy": 1.368529197246137,
   "spearman": 0.8989010989010989,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 60,
   "rank": 3,
   "margin": -1.1910980939865112,
   "entropy": 1.839948060908337,
   "spearman": 0.6835164835164835,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 61,
   "rank": 12,
   "margin": -5.39274275302887,
   "entropy": 1.7360616143454504,
   "spearman": 0.7450549450549451,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 62,
   "rank": 13,
   "margin": -6.692532181739807,
   "entropy": 1.6239995960281908,
   "spearman": 0.6791208791208792,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 63,
   "rank": 3,
   "margin": -1.289352297782898,
   "entropy": 1.6993898240717296,
   "spearman": 0.7406593406593407,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 64,
   "rank": 11,
   "margin": -5.874269008636475,
   "entropy": 1.741164670912533,
   "spearman": 0.6175824175824176,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 65,
   "rank": 9,
   "margin": -4.179586887359619,
   "entropy": 1.8288856658758939,
   "spearman": 0.8065934065934066,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 66,
   "rank": 8,
   "margin": -3.8839906454086304,
   "entropy": 1.7975342789154765,
   "spearman": 0.8945054945054945,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 67,
   "rank": 10,
   "margin": -4.605728268623352,
   "entropy": 1.7518785023346275,
   "spearman": 0.7054945054945055,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 68,
   "rank": 9,
   "margin": -5.00085186958313,
   "entropy": 1.5762649739069003,
   "spearman": 0.8373626373626374,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 69,
   "rank": 12,
   "margin": -5.89151656627655,
   "entropy": 1.6940678560908207,
   "spearman": 0.6659340659340659,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 70,
   "rank": 13,
   "margin": -7.423978090286255,
   "entropy": 1.6541884656896564,
   "spearman": 0.5868131868131868,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 71,
   "rank": 11,
   "margin": -6.2715067863464355,
   "entropy": 1.5379950488451422,
   "spearman": 0.7450549450549451,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 72,
   "rank": 4,
   "margin": -2.2368617057800293,
   "entropy": 1.4910532160671517,
   "spearman": 0.7450549450549451,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 73,
   "rank": 10,
   "margin": -5.5221551060676575,
   "entropy": 1.4124088790719886,
   "spearman": 0.7714285714285715,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 74,
   "rank": 11,
   "margin": -3.5235017140706386,
   "entropy": 1.8822352052192897,
   "spearman": 0.8593406593406593,
   "correct": "father",
   "pred": "nephew"
  },
  {
   "traj": 75,
   "rank": 13,
   "margin": -6.683761239051819,
   "entropy": 1.6555305803719884,
   "spearman": 0.6,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 76,
   "rank": 12,
   "margin": -6.0975006222724915,
   "entropy": 1.7003787515969426,
   "spearman": 0.6879120879120879,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 77,
   "rank": 12,
   "margin": -5.554402470588684,
   "entropy": 1.7861677683345922,
   "spearman": 0.701098901098901,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 78,
   "rank": 3,
   "margin": -1.2134548425674438,
   "entropy": 1.7654989227480482,
   "spearman": 0.7406593406593407,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 79,
   "rank": 11,
   "margin": -5.653041124343872,
   "entropy": 1.6573588600217926,
   "spearman": 0.8505494505494505,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 80,
   "rank": 7,
   "margin": -1.882861614227295,
   "entropy": 1.8406340254286608,
   "spearman": 0.6351648351648351,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 81,
   "rank": 10,
   "margin": -6.036560297012329,
   "entropy": 1.4070115789351842,
   "spearman": 0.8153846153846154,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 82,
   "rank": 13,
   "margin": -7.611722707748413,
   "entropy": 1.5159994454180077,
   "spearman": 0.5252747252747253,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 83,
   "rank": 9,
   "margin": -5.247303247451782,
   "entropy": 1.4662758997638683,
   "spearman": 0.9252747252747253,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 84,
   "rank": 6,
   "margin": -2.2837964296340942,
   "entropy": 1.6210186214940188,
   "spearman": 0.7054945054945055,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 85,
   "rank": 11,
   "margin": -5.170853316783905,
   "entropy": 1.7722146021056355,
   "spearman": 0.8725274725274725,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 86,
   "rank": 9,
   "margin": -4.951502323150635,
   "entropy": 1.559515680177243,
   "spearman": 0.8681318681318682,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 87,
   "rank": 13,
   "margin": -7.010478854179382,
   "entropy": 1.6523568427715196,
   "spearman": 0.6087912087912087,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 88,
   "rank": 10,
   "margin": -5.516029238700867,
   "entropy": 1.6572428328844884,
   "spearman": 0.8373626373626374,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 89,
   "rank": 10,
   "margin": -4.916640758514404,
   "entropy": 1.735888710589107,
   "spearman": 0.8197802197802198,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 90,
   "rank": 10,
   "margin": -4.820270299911499,
   "entropy": 1.623866299226988,
   "spearman": 0.7230769230769231,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 91,
   "rank": 10,
   "margin": -5.217213869094849,
   "entropy": 1.6386397096336924,
   "spearman": 0.7582417582417582,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 92,
   "rank": 11,
   "margin": -5.6928651332855225,
   "entropy": 1.803778010566667,
   "spearman": 0.7626373626373626,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 93,
   "rank": 12,
   "margin": -6.2361860275268555,
   "entropy": 1.59962979302855,
   "spearman": 0.5604395604395604,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 94,
   "rank": 13,
   "margin": -6.9854103326797485,
   "entropy": 1.527804890464957,
   "spearman": 0.5824175824175825,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 95,
   "rank": 10,
   "margin": -4.64037823677063,
   "entropy": 1.7635251919907766,
   "spearman": 0.8681318681318682,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 96,
   "rank": 6,
   "margin": -2.162471055984497,
   "entropy": 1.627800051761128,
   "spearman": 0.6879120879120879,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 97,
   "rank": 11,
   "margin": -5.270238637924194,
   "entropy": 1.6460313501066215,
   "spearman": 0.810989010989011,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 98,
   "rank": 12,
   "margin": -5.836217284202576,
   "entropy": 1.4380016155299293,
   "spearman": 0.6483516483516484,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 99,
   "rank": 13,
   "margin": -7.386413931846619,
   "entropy": 1.4605017799592774,
   "spearman": 0.6483516483516484,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 100,
   "rank": 13,
   "margin": -6.9505894780159,
   "entropy": 1.6333727360765993,
   "spearman": 0.6087912087912087,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 101,
   "rank": 9,
   "margin": -4.655502915382385,
   "entropy": 1.6790044439874625,
   "spearman": 0.8285714285714286,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 102,
   "rank": 11,
   "margin": -6.0046916007995605,
   "entropy": 1.5912495697049704,
   "spearman": 0.9076923076923077,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 103,
   "rank": 10,
   "margin": -5.372696876525879,
   "entropy": 1.6046342290278004,
   "spearman": 0.7582417582417582,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 104,
   "rank": 9,
   "margin": -4.3308329582214355,
   "entropy": 1.6569692693314888,
   "spearman": 0.8285714285714286,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 105,
   "rank": 10,
   "margin": -5.5203282833099365,
   "entropy": 1.4519953101465999,
   "spearman": 0.8021978021978022,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 106,
   "rank": 11,
   "margin": -5.7274569272994995,
   "entropy": 1.6686500078466049,
   "spearman": 0.7450549450549451,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 107,
   "rank": 11,
   "margin": -5.688892841339111,
   "entropy": 1.6092198028838056,
   "spearman": 0.7846153846153846,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 108,
   "rank": 5,
   "margin": -2.505741834640503,
   "entropy": 1.350976413651746,
   "spearman": 0.6967032967032967,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 109,
   "rank": 9,
   "margin": -4.95415198802948,
   "entropy": 1.6566551288369542,
   "spearman": 0.8857142857142857,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 110,
   "rank": 8,
   "margin": -4.3127923011779785,
   "entropy": 1.240831895008948,
   "spearman": 0.8725274725274725,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 111,
   "rank": 13,
   "margin": -7.395191013813019,
   "entropy": 1.3812044691865315,
   "spearman": 0.5604395604395604,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 112,
   "rank": 6,
   "margin": -2.0888811349868774,
   "entropy": 1.7818918698971615,
   "spearman": 0.7362637362637363,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 113,
   "rank": 11,
   "margin": -5.0409557819366455,
   "entropy": 1.6129273955861352,
   "spearman": 0.7934065934065934,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 114,
   "rank": 3,
   "margin": -1.2134548425674438,
   "entropy": 1.7654989227480482,
   "spearman": 0.7406593406593407,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 115,
   "rank": 8,
   "margin": -3.6981232166290283,
   "entropy": 1.608483998203236,
   "spearman": 0.8725274725274725,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 116,
   "rank": 11,
   "margin": -5.407118797302246,
   "entropy": 1.566975562775189,
   "spearman": 0.7582417582417582,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 117,
   "rank": 11,
   "margin": -5.471610188484192,
   "entropy": 1.6004495447945635,
   "spearman": 0.6131868131868132,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 118,
   "rank": 12,
   "margin": -5.800250172615051,
   "entropy": 1.4994017524070258,
   "spearman": 0.8637362637362638,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 119,
   "rank": 11,
   "margin": -6.705261468887329,
   "entropy": 1.1237985079602184,
   "spearman": 0.643956043956044,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 120,
   "rank": 12,
   "margin": -5.554402470588684,
   "entropy": 1.7861677683345922,
   "spearman": 0.701098901098901,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 121,
   "rank": 11,
   "margin": -5.841808319091797,
   "entropy": 1.5923017470961913,
   "spearman": 0.578021978021978,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 122,
   "rank": 12,
   "margin": -5.616938829421997,
   "entropy": 1.7758484423446672,
   "spearman": 0.6527472527472528,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 123,
   "rank": 13,
   "margin": -6.3660889863967896,
   "entropy": 1.6730584449140482,
   "spearman": 0.6263736263736264,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 124,
   "rank": 10,
   "margin": -4.9371771812438965,
   "entropy": 1.602160786147254,
   "spearman": 0.8461538461538461,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 125,
   "rank": 11,
   "margin": -5.648480772972107,
   "entropy": 1.8472828353949569,
   "spearman": 0.7758241758241758,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 126,
   "rank": 13,
   "margin": -6.385162115097046,
   "entropy": 1.6530458278440858,
   "spearman": 0.5560439560439561,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 127,
   "rank": 11,
   "margin": -6.109084606170654,
   "entropy": 1.6653072128554882,
   "spearman": 0.6351648351648351,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 128,
   "rank": 13,
   "margin": -7.1094189286231995,
   "entropy": 1.496283159899042,
   "spearman": 0.5032967032967033,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 129,
   "rank": 11,
   "margin": -4.450361251831055,
   "entropy": 1.7649082955365722,
   "spearman": 0.9076923076923077,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 130,
   "rank": 11,
   "margin": -5.2178261280059814,
   "entropy": 1.7214836840601526,
   "spearman": 0.5736263736263736,
   "correct": "daughter",
   "pred": "uncle"
  },
  {
   "traj": 131,
   "rank": 12,
   "margin": -6.482181429862976,
   "entropy": 1.4918587498028597,
   "spearman": 0.7494505494505495,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 132,
   "rank": 11,
   "margin": -5.566285967826843,
   "entropy": 1.2711943660078895,
   "spearman": 0.7846153846153846,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 133,
   "rank": 13,
   "margin": -7.3878209590911865,
   "entropy": 1.5653716448208101,
   "spearman": 0.6747252747252748,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 134,
   "rank": 6,
   "margin": -1.2548691034317017,
   "entropy": 1.8437330678564632,
   "spearman": 0.6967032967032967,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 135,
   "rank": 10,
   "margin": -4.336094856262207,
   "entropy": 1.6409496767960265,
   "spearman": 0.7098901098901099,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 136,
   "rank": 5,
   "margin": -1.4995365142822266,
   "entropy": 1.7636996405641434,
   "spearman": 0.6307692307692307,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 137,
   "rank": 5,
   "margin": -2.0499866604804993,
   "entropy": 1.4981028613311638,
   "spearman": 0.7450549450549451,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 138,
   "rank": 7,
   "margin": -2.4309157133102417,
   "entropy": 1.6502801750368312,
   "spearman": 0.6307692307692307,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 139,
   "rank": 4,
   "margin": -1.6714732646942139,
   "entropy": 1.6676707235955286,
   "spearman": 0.7406593406593407,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 140,
   "rank": 11,
   "margin": -5.250039577484131,
   "entropy": 1.545595592457588,
   "spearman": 0.8857142857142857,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 141,
   "rank": 5,
   "margin": -2.3905707597732544,
   "entropy": 1.5406796429861978,
   "spearman": 0.7098901098901099,
   "correct": "sister",
   "pred": "uncle"
  },
  {
   "traj": 142,
   "rank": 11,
   "margin": -4.9131019115448,
   "entropy": 1.7968175941310318,
   "spearman": 0.8373626373626374,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 143,
   "rank": 12,
   "margin": -6.139058947563171,
   "entropy": 1.5755922956937674,
   "spearman": 0.8417582417582418,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 144,
   "rank": 9,
   "margin": -4.394303143024445,
   "entropy": 1.7460934512300792,
   "spearman": 0.8021978021978022,
   "correct": "son",
   "pred": "uncle"
  },
  {
   "traj": 145,
   "rank": 10,
   "margin": -5.697374105453491,
   "entropy": 1.5484510181779432,
   "spearman": 0.832967032967033,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 146,
   "rank": 10,
   "margin": -4.186592102050781,
   "entropy": 1.838604693393202,
   "spearman": 0.810989010989011,
   "correct": "mother",
   "pred": "uncle"
  },
  {
   "traj": 147,
   "rank": 11,
   "margin": -5.083885192871094,
   "entropy": 1.7646376579130654,
   "spearman": 0.7978021978021979,
   "correct": "father",
   "pred": "uncle"
  },
  {
   "traj": 148,
   "rank": 13,
   "margin": -7.010478854179382,
   "entropy": 1.6523568427715196,
   "spearman": 0.6087912087912087,
   "correct": "brother",
   "pred": "uncle"
  },
  {
   "traj": 149,
   "rank": 11,
   "margin": -5.1647257804870605,
   "entropy": 1.7890790484518306,
   "spearman": 0.5868131868131868,
   "correct": "daughter",
   "pred": "uncle"
  }
 ]
}
```

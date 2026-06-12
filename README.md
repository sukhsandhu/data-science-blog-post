# Data Science Blog Post: What Really Drives Developer Salaries in 2024?

**Author:** Sukh Sandhu
**Date:** June 2026
**Udacity Data Scientist Nanodegree - Project 1**

[![GitHub Pages](https://img.shields.io/badge/Blog-GitHub%20Pages-blue)](https://sukhsandhu.github.io/data-science-blog-post/)

## Project Motivation

Developer pay is opaque: job ads hide ranges and anecdotes dominate. This project analyzes the **official 2024 StackOverflow Annual Developer Survey** (65,437 responses, 114 columns) using the CRISP-DM process to answer five business questions:

1. Which programming languages are associated with the highest salaries?
2. How do professional experience and education relate to compensation?
3. How does remote, hybrid or in-person work relate to job satisfaction?
4. How does compensation vary across countries?
5. Can we predict a developer's salary from their profile — and which factors matter most?

## Dataset

Official **StackOverflow Annual Developer Survey 2024** results.

> **Note:** the CSV (~150 MB) is **not committed** to this repository because of GitHub file-size limits. Download it from <https://survey.stackoverflow.co/> (2024 → *Data (CSV)*) and save it in the repository root as `so_survey_results_2024.csv` before running the notebook or script.

After filtering to full-time professional developers with a valid reported salary ($5k–$400k), the analysis uses **17,543 respondents**. Missing values are substantial in the raw survey (e.g. 64% of respondents skip compensation) — the notebook documents how every gap is handled (target rows dropped, nominal features given an explicit "Not specified" category, ordinal features encoded with a missing level, numeric experience median-imputed).

## Repository Structure

```
├── README.md                     <- This file
├── data_science_blog_post.ipynb  <- Jupyter notebook: full CRISP-DM analysis (report-style)
├── analysis.py                   <- Same pipeline as a standalone, documented script
├── index.md                      <- Blog post (GitHub Pages)
├── blog_post_summary.md          <- Plain-markdown copy of the blog post
├── requirements.txt              <- Python package requirements
└── salary_analysis.png           <- Four-panel results visualization
```

Run the script with: `python analysis.py [path/to/so_survey_results_2024.csv]`

## Libraries Used

- **pandas** — data manipulation and analysis
- **numpy** — numerical computing
- **scikit-learn** — preprocessing pipeline and Random Forest model
- **matplotlib** / **seaborn** — visualization
- **jupyter** — notebook environment

## Key Results

| Question | Finding |
|----------|---------|
| Q1 — Languages | Ruby ($95.0k) > Rust ($83.7k) > Go ($82.8k); Python mid-pack ($75.0k) |
| Q2 — Experience/Education | Experience r = 0.38; Bachelor's ($74.6k) out-earns Master's ($72.0k) at the median |
| Q3 — Work mode | Remote 7.1/10 satisfaction vs 6.7 in-person; salary↔satisfaction r ≈ 0.07 |
| Q4 — Geography | US ($145.0k) ≈ 7× India ($21.1k); country = 57% of model importance |
| Q5 — Model | Random Forest R² = 0.534, RMSE ≈ $43.3k on a 20% held-out test set |

**Example prediction:** US ML specialist (8 yrs, Master's, remote, 1,000–4,999-person org) → **~$150,000/year**

## Blog Post

Read the non-technical write-up: <https://sukhsandhu.github.io/data-science-blog-post/>

## Acknowledgments

- **StackOverflow** for publishing the annual developer survey data (<https://survey.stackoverflow.co/>)
- **Udacity** for the Data Scientist Nanodegree curriculum
- The open-source community behind pandas, scikit-learn and matplotlib

Survey data is used under StackOverflow's Open Database License (ODbL).

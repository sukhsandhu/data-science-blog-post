# Data Science Blog Post: What Really Drives Developer Salaries in 2024?

**Author:** Sukh Sandhu  
**Date:** June 2026  
**Udacity Data Scientist Nanodegree - Project 1**

[![GitHub Pages](https://img.shields.io/badge/Blog-GitHub%20Pages-blue)](https://sukhsandhu.github.io/data-science-blog-post/)

## Project Motivation

This project analyzes developer survey data to uncover key insights about compensation, job satisfaction, and career factors using the CRISP-DM process.

**Business Questions:**
1. What programming languages are most associated with higher salaries?
2. How do years of experience and education affect compensation?
3. What factors most strongly predict developer job satisfaction?
4. How does compensation vary across different countries?
5. Can we build a model to predict a developer's annual salary?

## Repository Structure

```
├── README.md                    <- This file
├── analysis.py                  <- Main Python analysis script (CRISP-DM)
├── index.md                     <- Blog post (GitHub Pages)
├── requirements.txt             <- Python package requirements
└── salary_analysis.png          <- Visualization (4 charts)
```

## Libraries Used

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning (Random Forest)
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

## Dataset

Simulated developer survey data based on StackOverflow Developer Survey patterns (2,000 records, 9 features: Country, Language, Education, DevType, YearsExp, CompanySize, RemoteWork, Salary, JobSatisfaction).

## Key Results

| Question | Finding |
|----------|---------|
| Q1 - Languages | Rust ($134k) > Scala ($127k) > Go ($118k) |
| Q2 - Experience | Correlation r=0.294; PhD earns $133k median |
| Q3 - Remote Work | Remote workers: 3.9/5 satisfaction vs 3.6/5 in-office |
| Q4 - Geography | US ($120k) vs India ($25k); Country = 68% feature importance |
| Q5 - Model | Random Forest R²=0.801, RMSE=$28,156 |

**Creative Prediction:** US ML Engineer (8 yrs, Rust, Master's, Large company, Remote) → **$221,283/year**

## Blog Post

Read the non-technical blog post at: https://sukhsandhu.github.io/data-science-blog-post/

## Acknowledgments

- StackOverflow for their annual developer survey
- Udacity for the Data Scientist Nanodegree curriculum
- The open-source community behind pandas, scikit-learn, and matplotlib

# 5632 Unsupervised Learning Final Project - Predicting Wine Ratings Using NLP

**Project Author:** Frederic G Miskawi  
**Course:** CSCA 5632 Unsupervised Learning Final Project  
**Professor:** Adjunct Professor Geena Kim - Online  
**Date:** 2025-11-30

---

## Executive Summary

**Problem:**  
Can wine quality scores be predicted from tasting notes alone? In my first class (supervised learning) final project, I predicted wine ratings using metadata like country, region, variety, and price—but left out the description text. Now, in this unsupervised learning project, I'm tackling an NLP-related challenge: can the *words* in a wine review tell us if a wine is good or outstanding?

**Why it matters:**  
Wineinformatics brings science to a field I always associated with subjective opinion. Understanding how language maps to quality scores could eventually power smarter wine recommendation apps—helping everyday consumers make informed decisions at the store.

---

## Methods

- **Unsupervised:** K-Means clustering on TF-IDF, SVD, and NMF features to discover natural groupings
- **Supervised:** Logistic Regression and Random Forest for binary classification (Good <90 vs Outstanding >=90)
- **Regression:** Gradient Boosting to predict actual point scores

---

## Data

~130k wine reviews from Wine Enthusiast (Kaggle dataset), focusing on the `description` and `designation` text columns.

**Dataset:** [Kaggle Wine Reviews (winemag-data-130k-v2)](https://www.kaggle.com/datasets/zynicide/wine-reviews)  
**Author:** zynicide  
**License:** CC BY-NC-SA 4.0

---

## Key Hypotheses

1. Wine scores can be predicted from text, but accuracy will be moderate
2. Binary categorization (Good vs Outstanding at 90 points) simplifies the problem
3. TF-IDF + supervised learning works well, even without semantic meaning/understanding
4. NMF + supervised learning should yield good results
5. Wine-specific terms correlate with high scores

---

## Summary of Findings

| Approach | Model | Result |
|----------|-------|--------|
| **Classification** | Logistic Regression (TF-IDF) | **~85% test accuracy** |
| **Regression** | Gradient Boosting | **R² = 0.67**, MAE ~1.4 points |
| **Unsupervised** | K-Means Clustering | ~52% accuracy (barely better than random) |

**Key Insight:** K-Means clustering grouped wines by *style/topics*, not *quality*—confirming that predicting quality requires labeled data. TF-IDF + Logistic Regression proved powerful (as it did in our BBC News project). Text features alone deliver strong results without needing price, region, or vintage.

---

## Future Considerations

- Doc2Vec, Word2Vec, or BERT embeddings to capture semantic meaning
- Sentiment analysis to normalize reviewer tendencies
- Combine text with price/region/vintage for improved predictions
- Build toward a Wine Recommender app in our upcoming deep learning class
- Test SVM with additional columns to build a more accurate model pipeline

---

## Repository Structure

```
├── 5632 Unsupervised Final Project - Wine Reviews.ipynb  # Main project notebook
├── README.md                                              # This file
└── winemag-data-130k-v2.csv                              # Dataset (download from Kaggle)
```

---

## Links

- **This Repo:** https://github.com/StLoupien/5632-Unsupervised-Learning---Final-Project
- **Video Presentation:** https://youtu.be/9X5zPXkQIxc
- **Previous Project (Supervised Learning):** https://github.com/StLoupien/CSCA-5622-Supervised-Learning-Final-Project---Frederic-Miskawi

---

## How to Run

1. Clone this repository
2. If kg import does not work for you, download the dataset from [Kaggle](https://www.kaggle.com/datasets/zynicide/wine-reviews) and place `winemag-data-130k-v2.csv` in the project folder. Update the path in the code cell.
3. Open `5632 Unsupervised Final Project - Wine Reviews.ipynb` in Jupyter Notebook or VS Code
4. Run all cells sequentially

**Requirements:** Python 3.x, scikit-learn, pandas, numpy, matplotlib, seaborn

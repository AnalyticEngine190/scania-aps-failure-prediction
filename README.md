# 🚛 Predictive Maintenance: Catching Truck Failures Before They Happen

## 📌 The TL;DR
I built a machine learning pipeline to predict when the Air Pressure System (APS) in Scania heavy trucks is about to fail, using a massive dataset of anonymized sensor telemetry. 

The twist? This wasn't about chasing standard "accuracy." In the real world, mechanics are cheap, but engine replacements are incredibly expensive. I had to design a **cost-sensitive** model that was mathematically "paranoid" enough to catch catastrophic breakdowns, even if it meant triggering a few false alarms along the way.

## ⚖️ The Challenge: Why 98% Accuracy is a Trap
In this dataset, actual APS failures are basically a needle in a haystack—they make up only **1.6%** of the data. If I built a lazy model that just guessed "healthy" every single time, it would be 98.4% accurate, but it would miss every single broken truck. 

Instead, I optimized the pipeline against a custom, real-world business cost metric:
* **Cost_1 (The False Alarm):** 10 units (Sending a mechanic to check a healthy truck)
* **Cost_2 (The Missed Breakdown):** 500 units (A truck dying on the side of the road)

My entire goal was to minimize that total financial penalty.

## ⚙️ How I Built It
1. **Cleaning the Mess:** The raw data was brutal—171 anonymized sensors with huge chunks of missing data. I used a most_frequent imputation strategy to fill the gaps so I didn't destroy the natural patterns of the truck's operational states. I choose most_frequent after testing imputation strategies.
2. **Cutting the Noise:** 171 sensors is a lot of noise. I trained a baseline XGBoost model purely to extract feature importances, which let me ruthlessly drop useless columns and isolate the **75 most critical sensors**.
3. **Tuning the Engine:** With a lean dataset, I brought in **LightGBM**. To handle the massive class imbalance, I cranked up the `scale_pos_weight` to 59 (forcing the algorithm to care 59x more about failures) and tuned the hyperparameters using RandomizedSearchCV.
4. **The Game-Changer (Threshold Moving):** By default, ML models only flag a failure if they are >50% sure. I ripped out the raw prediction probabilities and dragged the decision boundary down to **30%**. This made the model highly sensitive, sacrificing a few cheap false positives to catch almost all of the expensive false negatives.

## 🏆 The Final Exam (Holdout Test Set)
I tested the final LightGBM pipeline on a completely unseen holdout set of 16,000 trucks. Here is what happened:
* **False Positives:** 467
* **False Negatives:** Just 15
* **Total Cost Penalty:** 12,170 

*Context: This exact dataset was used for the 2016 IDA Industrial Challenge. My pipeline's custom score of 12,170 effectively ties for a **4th-place global finish** on that original professional leaderboard.*

## 🛠️ The Tech Stack
* **Tools:** Python, Jupyter Notebooks, PyCharm
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Joblib

---
This has been a massive experience for me and I had a lot of fun doing this project. I hope you also had fun reading it.
*I love building things that solve complex, messy data problems. Right now, I'm working on concurrent projects involving logistics data analysis and network security threat detection (ISCX).*

# 🚛 Predictive Maintenance: Catching Truck Failures Before They Happen
**Author:** Serhat Turan  

👉 **[Live Interactive Dashboard: Try the Fleet Maintenance Predictor Here](https://scania-aps-failure-prediction-g25sfbgnhe4ecjjzcmbvnz.streamlit.app)**

### 📌 The TL;DR
I built a machine learning pipeline to predict when the Air Pressure System (APS) in Scania heavy trucks is about to fail, using a massive dataset of anonymized sensor telemetry. 

The twist? This wasn't about chasing standard "accuracy." In the real world, mechanics are cheap, but engine replacements are incredibly expensive. I had to design a cost-sensitive model that was mathematically "paranoid" enough to catch catastrophic breakdowns, even if it meant triggering a few false alarms along the way.

### ⚖️ The Challenge: Why 98% Accuracy is a Trap
In this dataset, actual APS failures are basically a needle in a haystack—they make up only 1.6% of the data. If I built a lazy model that just guessed "healthy" every single time, it would be 98.4% accurate, but it would miss every single broken truck.

Instead, I optimized the pipeline against a custom, real-world business cost metric:
*   **Cost_1 (The False Alarm):** 10 units (Sending a mechanic to check a healthy truck)
*   **Cost_2 (The Missed Breakdown):** 500 units (A truck dying on the side of the road)

My entire goal was to minimize that total financial penalty.

### ⚙️ How I Built It
*   **Cleaning the Mess:** The raw data was brutal—171 anonymized sensors with huge chunks of missing data. After testing multiple strategies, I implemented a `most_frequent` imputation pipeline to fill gaps without destroying the natural patterns of the truck's operational states.
*   **Dodging the Data Leakage Trap:** Initially, my baseline model looked a little *too* good. I realized I was imputing missing values before splitting my train/test sets—a classic data leakage error. I refactored the pipeline to strictly fit the imputer on the training data only, ensuring my test results reflected true real-world robustness. 
*   **Cutting the Noise:** 171 sensors is a lot of noise. I trained a baseline XGBoost model purely to extract feature importances, ruthlessly dropping useless columns to isolate the 75 most critical sensors.
*   **Tuning the Engine:** With a lean dataset, I brought in LightGBM. To handle the massive class imbalance, I cranked up the `scale_pos_weight` to 59 (forcing the algorithm to care 59x more about failures) and tuned the hyperparameters using `RandomizedSearchCV`.
*   **Threshold Moving:** I extracted the raw prediction probabilities to find the absolute most cost-effective decision boundary, finding the perfect balance between cheap false positives and expensive false negatives.

### 🏆 The Final Exam (Holdout Test Set)
I tested the final LightGBM pipeline on a completely unseen holdout set of 16,000 trucks. Here is what happened:
*   **False Positives:** 550
*   **False Negatives:** 17
*   **Total Cost Penalty:** 14,000

*Context: This exact dataset was used for the 2016 IDA Industrial Challenge. My pipeline's custom score represents a highly competitive, production-ready finish compared to the original professional leaderboard.*

### 🛠️ The Tech Stack
*   **Environment:** Python, Jupyter Notebooks, PyCharm
*   **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Joblib, Streamlit

### 👨‍💻 About Me
I love building things that solve complex, messy data problems for the heavy equipment manufacturing sector—especially optimizing predictive maintenance pipelines for industrial fleets. Having graduated with a B.Sc. in Mathematics from Dokuz Eylül University, I am currently based in İzmir, Turkey. With a strong foundation in statistical modeling and a B2-level proficiency in Swedish, I am actively seeking a Junior Data Scientist role where I can apply my skills in machine learning to high-impact industrial data challenges. Alongside this project, I am currently developing concurrent analytical projects involving logistics data analysis and ISCX network security threat detection.

# **Trader Behavior Insights – Assignment Submission**

This project is part of the **Junior Data Scientist – Trader Behavior Insights** hiring process.
The goal is to explore how trader performance changes with shifts in market sentiment, especially during periods of Fear and Greed in the Bitcoin ecosystem.

---

## **📌 Assignment Overview**

The task involves working with two datasets:

### **1. Bitcoin Market Sentiment (Fear & Greed Index)**

**Columns:**

* Date
* Classification (Fear / Greed)

### **2. Hyperliquid Historical Trader Data**

**Columns include:**

* account
* symbol
* execution price
* size
* side
* time
* start position
* event
* closedPnL
* leverage
* and other related trading fields

---

## **🎯 Objective of the Analysis**

* Understand how traders behave in different sentiment conditions
* Compare profitability (PnL), risk-taking, and trade frequency across Fear vs. Greed periods
* Identify any consistent patterns that can support smarter trading strategies
* Explore trader behavior on Hyperliquid during extreme sentiment zones
* Build clear, data-driven insights backed by visualizations and simple metrics

---

## **📁 Dataset Links**

* **Hyperliquid Trader Data:**
  [https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing](https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view?usp=sharing)

* **Bitcoin Fear & Greed Index:**
  [https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing](https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view?usp=sharing)

---

## **🛠️ Approach Used in the Assignment**

1. **Data Loading & Cleaning**

   * Parsed timestamps
   * Handled missing or inconsistent values
   * Standardized column names

2. **Feature Engineering**

   * Merged datasets on date/time
   * Categorized trading behavior under Fear and Greed
   * Computed simple metrics such as:

     * Average PnL
     * Position size trends
     * Leverage usage
     * Win/loss distribution

3. **Exploratory Analysis**

   * Trend charts for PnL vs. sentiment
   * Distribution of trades in different sentiment phases
   * Comparison of trader aggression (size, leverage)
   * Identification of unusual patterns or anomalies

4. **Insights & Observations**

   * Key findings are summarized in the notebook/analysis section
   * Includes practical interpretations useful for trading strategy teams

---

## **📦 Repository Structure**

```
├── data/               # Raw datasets (if allowed)
├── notebooks/          # Jupyter notebook(s) with full analysis
├── scripts/            # Helper Python scripts (if any)
├── visuals/            # Plots generated during analysis
└── README.md           # Project overview (this file)
```

---

## **📬 Submission Details**

As instructed, the final submission (GitHub link + resume) should be emailed to:

* [saami@bajarangs.com](mailto:saami@bajarangs.com)
* [nagasai@bajarangs.com](mailto:nagasai@bajarangs.com)
* [chetan@bajarangs.com](mailto:chetan@bajarangs.com)
  **CC:** [sonika@primetrade.ai](mailto:sonika@primetrade.ai)

**Subject:**
`Junior Data Scientist – Trader Behavior Insights`

---

## **⏳ Timeline**

* Application Deadline: *[Two weeks from posting]*
* Shortlisted candidates will be notified within **3 business days**
* Early submissions are given priority (“First come, first serve”)

---

## **👤 Ideal Candidate Profile**

* Recent graduates
* Bootcamp graduates
* Undergraduates
* Crypto-native analysts with strong analytical skills

---

## **🚀 Final Note**

This assignment is a great opportunity to demonstrate practical problem-solving, clarity of thought, and data intuition.
Thank you for reviewing my submission — I look forward to the next steps.

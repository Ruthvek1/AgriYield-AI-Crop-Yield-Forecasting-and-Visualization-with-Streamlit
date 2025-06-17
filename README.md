# AgriYield AI: Crop Yield Forecasting and Visualization with Streamlit

This project showcases a Machine Learning-based solution for predicting agricultural crop yield using a rich dataset and visualizing the results in a user-friendly **Streamlit web app**. It helps stakeholders—farmers, researchers, and policymakers—make data-driven decisions for agricultural planning and sustainability.

---

## 🌱 Project Objectives

- Predict crop yield based on historical agricultural data.
- Provide interactive visualizations for crop data across Indian states.
- Deliver an intuitive Streamlit interface for real-time predictions and exploration.

---

## 🧰 Tools & Technologies Used

- **Frontend**: Streamlit, HTML, JS, GeoJSON
- **Backend**: Python, Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly, Streamlit components, GeoJSON maps
- **ML Model**: Regression (using Scikit-learn)
- **Deployment**: Local Streamlit Server

---

## 📊 Dataset

- **File**: `crop_yield.csv`
- **Attributes**: Crop, State, Area, Production, Rainfall, Fertilizer, Pesticide, Year, Season
- **Size**: ~19,000 records
- **Target Variable**: `Yield`

---

## 🧠 Methodology

1. **Data Cleaning & Preprocessing**
   - Removed nulls, encoded categories, normalized features.

2. **Feature Engineering**
   - Created new attributes from seasonal and geographical data.

3. **Modeling**
   - Trained a regression model using Scikit-learn.

4. **Visualization**
   - Interactive choropleth maps and input forms via Streamlit.

5. **Deployment**
   - Integrated everything into a working web application (`app.py`).

---

## 🛠 How to Run the Project

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/AgriYield-AI.git
   cd AgriYield-AI

# # anomaly_app.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# from joblib import load

# st.title("Anomaly Detection App")

# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     st.write("Data Preview:", data.head())

#     if 'Time' in data.columns:
#         data = data.drop('Time', axis=1)

#     model = load("xgb_model.joblib")  # Make sure this file exists in the same folder
#     prediction = model.predict(data)
#     st.write("Predictions:")
#     st.write(prediction)





import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

# Page title
st.title("🔍 Anomaly Detection App")

# Upload CSV file
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

# Process after upload
if uploaded_file is not None:
    st.subheader("📑 Step 1: Preview Uploaded Data")
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Optional: Drop unwanted columns like 'Time' if present
    if 'Time' in data.columns:
        data = data.drop('Time', axis=1)

    try:
        # Load trained model
        model = load("xgb_model.joblib")  # Ensure this file is in same directory

        # Load scaler if needed
        try:
            scaler = load("scaler.joblib")
            data_scaled = scaler.transform(data)
        except:
            data_scaled = data  # Use raw data if no scaler available

        # Prediction
        prediction = model.predict(data_scaled)
        data['Prediction'] = prediction

        st.subheader("✅ Step 2: Prediction Completed")
        st.write(data[['Prediction']].value_counts().rename("Count").reset_index())

        # Bar Chart
        st.subheader("📊 Step 3: Prediction Distribution")
        fig, ax = plt.subplots()
        pd.Series(prediction).value_counts().sort_index().plot(kind='bar', color=['green', 'red'], ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal (0)', 'Anomaly (1)'])
        ax.set_ylabel('Count')
        ax.set_title('Anomaly vs Normal Prediction Count')
        st.pyplot(fig)

        # Download Predictions
        st.subheader("⬇️ Step 4: Download Predictions as CSV")
        csv_data = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV File with Predictions",
            data=csv_data,
            file_name="predicted_output.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

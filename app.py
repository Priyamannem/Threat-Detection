import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Cybersecurity Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load the model and preprocessing objects
@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model, scaler, and other necessary objects"""
    try:
        # Load the Random Forest model
        with open('random_forest_model.pkl', 'rb') as file:
            model = pickle.load(file)
        
        # Load the scaler (you need to save this during training)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        # Load label mapping (you need to save this during training)
        with open('label_mapping.pkl', 'rb') as file:
            label_mapping = pickle.load(file)
        
        # Load training features (column names after encoding)
        with open('feature_names.pkl', 'rb') as file:
            feature_names = pickle.load(file)
        
        return model, scaler, label_mapping, feature_names
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.info("Please ensure all required pickle files are in the same directory.")
        return None, None, None, None

# Load model and preprocessors
model, scaler, label_mapping, feature_names = load_model_and_preprocessors()

# Reverse label mapping for predictions
if label_mapping:
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Title and description
st.title("🛡️ Cybersecurity Threat Detection System")
st.markdown("""
This application uses a **Random Forest Classifier** to detect and classify cybersecurity threats 
based on network traffic patterns and cryptocurrency transaction data.
""")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Single Prediction", "Batch Prediction", "Model Info"])

# ==================== SINGLE PREDICTION PAGE ====================
if page == "Single Prediction":
    st.header("🔍 Single Threat Detection")
    st.markdown("Enter the network traffic details below to detect potential threats.")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time = st.number_input("Time", min_value=0, max_value=100, value=50, 
                                   help="Time interval of network activity")
            protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"], 
                                    help="Network protocol type")
            flag = st.selectbox("Flag", ["A", "S", "F", "R", "P"], 
                                help="TCP flag indicator")
            family = st.selectbox("Malware Family", 
                                  ["WannaCry", "Mirai", "Zeus", "Emotet", "TrickBot"],
                                  help="Known malware family")
            clusters = st.number_input("Cluster", min_value=1, max_value=10, value=1,
                                       help="Cluster identifier")
        
        with col2:
            send_address = st.text_input("Sender Address", value="1DA11mPS",
                                        help="Source Bitcoin wallet address")
            exp_address = st.text_input("Export Address", value="1BonuSr7",
                                       help="Destination Bitcoin wallet address")
            btc = st.number_input("BTC Amount", min_value=0.0, max_value=100.0, 
                                  value=1.0, step=0.1,
                                  help="Bitcoin transaction amount")
            usd = st.number_input("USD Value", min_value=0, max_value=100000, 
                                  value=500,
                                  help="USD equivalent of transaction")
        
        with col3:
            netflow_bytes = st.number_input("Netflow Bytes", min_value=0, 
                                           max_value=10000, value=5,
                                           help="Number of bytes in network flow")
            ip_address = st.text_input("IP Address Category", value="A",
                                      help="IP address classification")
            threats = st.selectbox("Threat Type", 
                                  ["Bonet", "Ransomware", "DDoS", "Phishing", "Malware"],
                                  help="Type of detected threat")
            port = st.number_input("Port", min_value=0, max_value=65535, 
                                   value=5061,
                                   help="Network port number")
        
        # Submit button
        submitted = st.form_submit_button("🔎 Detect Threat", use_container_width=True)
    
    if submitted and model is not None:
        # Create input dataframe
        user_input = pd.DataFrame({
            'Time': [time],
            'Protcol': [protocol],
            'Flag': [flag],
            'Family': [family],
            'Clusters': [clusters],
            'SeddAddress': [send_address],
            'ExpAddress': [exp_address],
            'BTC': [btc],
            'USD': [usd],
            'Netflow_Bytes': [netflow_bytes],
            'IPaddress': [ip_address],
            'Threats': [threats],
            'Port': [port]
        })
        
        try:
            # Preprocess input
            categorical_cols = user_input.select_dtypes(include=['object']).columns
            user_encoded = pd.get_dummies(user_input, columns=categorical_cols, drop_first=True)
            
            # Align with training features
            user_aligned = user_encoded.reindex(columns=feature_names, fill_value=0)
            
            # Scale features
            user_scaled = scaler.transform(user_aligned.values)
            
            # Make prediction
            prediction_numeric = model.predict(user_scaled)
            prediction_proba = model.predict_proba(user_scaled)
            
            # Get predicted label
            predicted_label = reverse_label_mapping[prediction_numeric[0]]
            confidence = np.max(prediction_proba) * 100
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", predicted_label)
            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")
            with col3:
                threat_level = "🔴 HIGH" if confidence > 90 else "🟡 MEDIUM" if confidence > 70 else "🟢 LOW"
                st.metric("Threat Level", threat_level)
            
            # Show probability distribution
            st.subheader("Probability Distribution")
            proba_df = pd.DataFrame({
                'Class': [reverse_label_mapping[i] for i in range(len(prediction_proba[0]))],
                'Probability': prediction_proba[0] * 100
            }).sort_values('Probability', ascending=False)
            
            st.bar_chart(proba_df.set_index('Class'))
            
            # Detailed results
            with st.expander("📊 View Detailed Results"):
                st.dataframe(proba_df, use_container_width=True)
                st.json({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": predicted_label,
                    "confidence": f"{confidence:.2f}%",
                    "input_features": user_input.to_dict('records')[0]
                })
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please check if all preprocessing files are loaded correctly.")

# ==================== BATCH PREDICTION PAGE ====================
elif page == "Batch Prediction":
    st.header("📁 Batch Threat Detection")
    st.markdown("Upload a CSV file with multiple records for batch prediction.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_data = pd.read_csv(uploaded_file)
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(batch_data.head(10), use_container_width=True)
            
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    # Preprocess batch data
                    categorical_cols = batch_data.select_dtypes(include=['object']).columns
                    batch_encoded = pd.get_dummies(batch_data, columns=categorical_cols, drop_first=True)
                    batch_aligned = batch_encoded.reindex(columns=feature_names, fill_value=0)
                    batch_scaled = scaler.transform(batch_aligned.values)
                    
                    # Make predictions
                    predictions_numeric = model.predict(batch_scaled)
                    predictions_proba = model.predict_proba(batch_scaled)
                    
                    # Add predictions to dataframe
                    batch_data['Predicted_Class'] = [reverse_label_mapping[p] for p in predictions_numeric]
                    batch_data['Confidence'] = np.max(predictions_proba, axis=1) * 100
                    
                    st.success(f"✅ Processed {len(batch_data)} records successfully!")
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(batch_data, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Prediction Distribution")
                        pred_counts = batch_data['Predicted_Class'].value_counts()
                        st.bar_chart(pred_counts)
                    
                    with col2:
                        st.subheader("Summary Statistics")
                        st.metric("Total Records", len(batch_data))
                        st.metric("Average Confidence", f"{batch_data['Confidence'].mean():.2f}%")
                        st.metric("Most Common Threat", pred_counts.index[0])
                    
                    # Download results
                    csv = batch_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"threat_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ==================== MODEL INFO PAGE ====================
elif page == "Model Info":
    st.header("ℹ️ Model Information")
    
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.write(f"**Model Type:** Random Forest Classifier")
            st.write(f"**Number of Trees:** {model.n_estimators}")
            st.write(f"**Number of Features:** {len(feature_names)}")
            st.write(f"**Number of Classes:** {len(label_mapping)}")
        
        with col2:
            st.subheader("Performance Metrics")
            st.write("**Accuracy:** 99.35%")
            st.write("**Precision:** 99.35%")
            st.write("**Recall:** 99.35%")
            st.info("Metrics calculated on test dataset")
        
        st.subheader("Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            st.bar_chart(importance_df.set_index('Feature'))
        
        st.subheader("Target Classes")
        st.write(f"**Classes:** {', '.join(reverse_label_mapping.values())}")
        
        with st.expander("📚 About the Dataset"):
            st.markdown("""
            This model is trained on cybersecurity threat data containing:
            - Network traffic patterns
            - Cryptocurrency transaction information
            - Malware family classifications
            - Port and protocol information
            - Various network flow metrics
            """)
    else:
        st.warning("Model not loaded. Please check your pickle files.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🛡️ Cybersecurity Threat Detection System | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
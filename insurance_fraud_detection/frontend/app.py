import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import warnings
import os # Import os to get environment variable

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚗 Insurance Fraud Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 2rem; }
    .prediction-container { padding: 2rem; border-radius: 15px; color: white; text-align: center; margin: 2rem 0; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
    .fraud-detected { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); border: 3px solid #ff4757; }
    .no-fraud-detected { background: linear-gradient(135deg, #00b894 0%, #00a085 100%); border: 3px solid #00cec9; }
    .metric-card { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.5rem; border-radius: 15px; border-left: 5px solid #667eea; margin: 0.5rem 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1); transition: transform 0.3s ease; }
    .metric-card:hover { transform: translateY(-5px); }
    .sidebar-header { font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin-bottom: 1rem; text-align: center; }
    .risk-factor { background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; }
    .safe-factor { background: #d1ecf1; border: 1px solid #74b9ff; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; }
    .stButton > button { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 25px; padding: 0.75rem 2rem; font-weight: bold; font-size: 1.1rem; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
</style>
""", unsafe_allow_html=True)

# Get FastAPI URL from environment variable, default to localhost for local testing
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

@st.cache_data
def load_sample_data():
    try:
        # Assuming insurance_claims.csv is copied into the frontend container's /app directory
        df = pd.read_csv('insurance_claims.csv')
        return df
    except FileNotFoundError:
        st.warning("⚠️ Sample data file not found in frontend container. Using default thresholds for display purposes.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error loading sample data in frontend: {str(e)}")
        return None

def create_gauge_chart(confidence, title):
    color = "red" if confidence > 0.7 else "orange" if confidence > 0.5 else "green"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "#d4edda"},
                {'range': [50, 80], 'color': "#fff3cd"},
                {'range': [80, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
    return fig

def create_comparison_chart(comparison_metrics):
    df_comparison = pd.DataFrame([
        {
            'Metric': key.replace('_', ' ').title(),
            'User Value': value['user_value'],
            'Industry Average': value['industry_avg']
        } for key, value in comparison_metrics.items()
    ])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Your Claim',
        x=df_comparison['Metric'],
        y=df_comparison['User Value'],
        marker_color='#667eea'
    ))
    fig.add_trace(go.Bar(
        name='Industry Average',
        x=df_comparison['Metric'],
        y=df_comparison['Industry Average'],
        marker_color='#ff6b6b'
    ))
    fig.update_layout(
        title='Claim vs Industry Averages',
        barmode='group',
        height=400,
        yaxis_title='Value'
    )
    return fig

def main():
    st.markdown('<div class="main-header">🚗 Insurance Fraud Detection System</div>', unsafe_allow_html=True)
    
    sample_df = load_sample_data()
    
    tab1, tab2, tab3 = st.tabs(["🔍 Fraud Detection", "📊 Model Information", "📈 Analytics Dashboard"])
    
    with tab1:
        st.sidebar.markdown('<div class="sidebar-header">📝 Enter Claim Details</div>', unsafe_allow_html=True)
        
        with st.sidebar.expander("👤 Customer Information", expanded=True):
            months_as_customer = st.number_input("Months as Customer", min_value=1, max_value=500, value=100)
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
            # FIX: Ensure min_value and max_value are floats for float 'value'
            policy_annual_premium = st.number_input("Policy Annual Premium ($)", min_value=100.0, max_value=5000.0, value=1200.0)
            umbrella_limit = st.selectbox("Umbrella Limit ($)", [0, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000])
        
        with st.sidebar.expander("📋 Policy Information", expanded=True):
            policy_bind_date = st.date_input("Policy Bind Date", date(2015, 1, 1))
            policy_state = st.selectbox("Policy State", ["OH", "IN", "IL", "WV", "VA", "NY", "SC", "NC"])
        
        with st.sidebar.expander("🚨 Incident Details", expanded=True):
            incident_type = st.selectbox("Incident Type", ["Multi-vehicle Collision", "Single Vehicle Collision", "Vehicle Theft", "Parked Car"])
            collision_type = st.selectbox("Collision Type", ["Rear Collision", "Front Collision", "Side Collision", "?"])
            incident_severity = st.selectbox("Incident Severity", ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"])
            incident_hour_of_the_day = st.slider("Incident Hour of Day", 0, 23, 12)
            number_of_vehicles_involved = st.number_input("Number of Vehicles Involved", min_value=1, max_value=10, value=1)
            witnesses = st.number_input("Number of Witnesses", min_value=0, max_value=10, value=2)
        
        with st.sidebar.expander("💰 Claim Details", expanded=True):
            # FIX: Ensure min_value and max_value are floats for float 'value'
            total_claim_amount = st.number_input("Total Claim Amount ($)", min_value=100.0, max_value=200000.0, value=20000.0)
            col1, col2 = st.columns(2)
            with col1:
                # FIX: Ensure min_value and max_value are floats for float 'value'
                injury_claim = st.number_input("Injury Claim ($)", min_value=0.0, max_value=total_claim_amount, value=min(2000.0, total_claim_amount//4))
            with col2:
                # FIX: Ensure min_value and max_value are floats for float 'value'
                property_claim = st.number_input("Property Claim ($)", min_value=0.0, max_value=total_claim_amount, value=min(5000.0, total_claim_amount//4))
            remaining_amount = max(0.0, total_claim_amount - injury_claim - property_claim)
            # FIX: Ensure min_value and max_value are floats for float 'value'
            vehicle_claim = st.number_input("Vehicle Claim ($)", min_value=0.0, max_value=total_claim_amount, value=min(remaining_amount, total_claim_amount))
            total_individual = injury_claim + property_claim + vehicle_claim
            if total_individual != total_claim_amount:
                st.warning(f"⚠️ Individual claims ({total_individual:,.0f}) don't sum to total claim amount ({total_claim_amount:,.0f})")
        
        with st.sidebar.expander("🚗 Vehicle Information", expanded=True):
            auto_model = st.selectbox("Auto Model", ["Camry", "Civic", "Accord", "Corolla", "F150", "Malibu", "Altima", "Sentra", "Wrangler", "Cherokee", "Forester", "Outback", "A4", "BMW 3 Series", "C300", "Tahoe", "Pathfinder", "92x", "95", "E400", "RAM", "RSX", "A5", "Other"])
            auto_year = st.number_input("Auto Year", min_value=1990, max_value=datetime.now().year, value=2010)
        
        with st.sidebar.expander("👨‍💼 Personal Details", expanded=True):
            insured_hobbies = st.selectbox("Insured Hobbies", ["reading", "board-games", "hiking", "camping", "golf", "base-jumping", "bungie-jumping", "sleeping", "cross-fit", "kayaking", "polo", "dancing", "chess", "football", "basketball", "skydiving", "craft-repair", "machine-op-inspct"])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔍 Fraud Detection Analysis")
            
            if st.button("🚀 Analyze Claim for Fraud", type="primary"):
                with st.spinner("Analyzing claim data..."):
                    # CRITICAL FIX: Pass the actual variable values, not their names as strings
                    input_data = {
                        'months_as_customer': months_as_customer,
                        'age': age,
                        'policy_annual_premium': policy_annual_premium,
                        'umbrella_limit': umbrella_limit,
                        'policy_bind_date': policy_bind_date.isoformat(), # Convert date object to "YYYY-MM-DD" string
                        'policy_state': policy_state,
                        'insured_hobbies': insured_hobbies,
                        'incident_type': incident_type,
                        'collision_type': collision_type,
                        'incident_severity': incident_severity,
                        'incident_hour_of_the_day': incident_hour_of_the_day,
                        'number_of_vehicles_involved': number_of_vehicles_involved,
                        'witnesses': witnesses,
                        'total_claim_amount': total_claim_amount,
                        'injury_claim': injury_claim,
                        'property_claim': property_claim,
                        'vehicle_claim': vehicle_claim,
                        'auto_model': auto_model,
                        'auto_year': auto_year
                    }
                  
                    try:
                        # Use the FASTAPI_URL environment variable here
                        response = requests.post(f"{FASTAPI_URL}/predict", json=input_data)
                        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                        result = response.json()
                        
                        prediction = result['prediction']
                        fraud_probability = result['fraud_probability']
                        no_fraud_probability = result['no_fraud_probability']
                        risk_factors = result['risk_factors']
                        safe_factors = result['safe_factors']
                        comparison_metrics = result['comparison_metrics']
                        
                        if prediction == 1:
                            st.markdown(f"""
                            <div class="prediction-container fraud-detected">
                                <h1>⚠️ FRAUD ALERT</h1>
                                <h2>High Risk Detection</h2>
                                <h3>Fraud Confidence: {fraud_probability:.1%}</h3>
                                <p style="font-size: 1.2em;">This claim exhibits multiple fraud indicators and requires immediate investigation</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-container no-fraud-detected">
                                <h1>✅ CLAIM APPROVED</h1>
                                <h2>Low Risk Detection</h2>
                                <h3>Legitimate Confidence: {no_fraud_probability:.1%}</h3>
                                <p style="font-size: 1.2em;">This claim appears legitimate and can proceed with standard processing</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.subheader("📈 Probability Analysis")
                        col_prob1, col_prob2 = st.columns(2)
                        with col_prob1:
                            gauge_fig = create_gauge_chart(fraud_probability, "Fraud Risk Score")
                            st.plotly_chart(gauge_fig, use_container_width=True)
                        with col_prob2:
                            prob_df = pd.DataFrame({
                                'Outcome': ['Legitimate', 'Fraudulent'],
                                'Probability': [no_fraud_probability, fraud_probability]
                            })
                            fig_prob = px.pie(prob_df, values='Probability', names='Outcome',
                                             color_discrete_map={'Legitimate': '#00b894', 'Fraudulent': '#ff6b6b'},
                                             title="Risk Distribution")
                            st.plotly_chart(fig_prob, use_container_width=True)
                        
                        st.subheader("🎯 Risk Factor Analysis")
                        col_risk1, col_risk2 = st.columns(2)
                        with col_risk1:
                            st.markdown("#### ⚠️ Risk Factors")
                            if risk_factors:
                                for factor in risk_factors:
                                    severity_icon = "🔴" if factor.get('severity') == 'high' else "🟡" if factor.get('severity') == 'medium' else "🟠"
                                    st.markdown(f"""
                                    <div class="risk-factor" style="color: black;">
                                        <strong style="color: black;">{severity_icon} {factor['factor']}</strong><br>
                                        <small style="color: black;">{factor['description']}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.success("✅ No significant risk factors identified")
                        with col_risk2:
                            st.markdown("#### ✅ Positive Factors")
                            if safe_factors:
                                for factor in safe_factors:
                                    st.markdown(f"""
                                    <div class="safe-factor" style="color: black;">
                                        <strong style="color: black;">✅ {factor['factor']}</strong><br>
                                        <small style="color: black;">{factor['description']}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No specific positive factors identified")
                        
                        st.subheader("📊 Industry Comparison")
                        comparison_fig = create_comparison_chart(comparison_metrics)
                        st.plotly_chart(comparison_fig, use_container_width=True)
                        
                        st.subheader("💡 Recommendations")
                        if prediction == 1:
                            st.error("""
                            **Immediate Actions Required:**
                            - 🔍 Conduct thorough manual review
                            - 📞 Contact customer for additional information
                            - 🕵️ Investigate claim circumstances
                            - 📋 Request additional documentation
                            - ⏸️ Hold payment pending investigation
                            """)
                        else:
                            if fraud_probability > 0.3:
                                st.warning("""
                                **Recommended Actions:**
                                - 👀 Standard verification process
                                - 📄 Review documentation completeness
                                - ✅ Proceed with normal processing timeline
                                """)
                            else:
                                st.success("""
                                **Recommended Actions:**
                                - ✅ Fast-track for approval
                                - 📋 Standard documentation review
                                - 💰 Proceed with payment processing
                                """)
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Error communicating with backend: {str(e)}. Make sure backend is running and accessible at {FASTAPI_URL}")
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
        
        with col2:
            st.subheader("📊 Real-time Metrics")
            st.markdown("### 🎯 Model Confidence")
            confidence_placeholder = st.empty() # This placeholder is not actively updated in the current code
            st.markdown("### 📈 Quick Stats")
            if sample_df is not None:
                stats = {
                    'avg_claim': sample_df['total_claim_amount'].mean(),
                    'claim_75th': sample_df['total_claim_amount'].quantile(0.75),
                    'avg_age': sample_df['age'].mean(),
                    'avg_premium': sample_df['policy_annual_premium'].mean()
                }
                st.metric("Avg Claim Amount", f"${stats['avg_claim']:,.0f}")
                st.metric("Fraud Threshold", f"${stats['claim_75th']:,.0f}")
                st.metric("Avg Customer Age", f"{stats['avg_age']:.0f} years")
                st.metric("Avg Premium", f"${stats['avg_premium']:,.0f}")
            else:
                st.info("Load sample data to view quick stats.")
    
    with tab2:
        st.header("📊 Model Information & Performance")
        st.info("This section is a placeholder. You can expand it to fetch model details from the backend or display static information about your model's performance metrics (e.g., accuracy, precision, recall).")
        if sample_df is not None:
            # Example: Display basic info if 'fraud_reported' column exists
            if 'fraud_reported' in sample_df.columns:
                fraud_count = (sample_df['fraud_reported'] == 'Y').sum()
                total_count = len(sample_df)
                st.write(f"Dataset Size: {total_count} records")
                st.write(f"Number of Fraudulent Claims: {fraud_count}")
                st.write(f"Fraud Percentage: {(fraud_count / total_count) * 100:.2f}%")
        else:
            st.warning("Sample data not available to show model information.")
    
    with tab3:
        st.header("📈 Analytics Dashboard")
        if sample_df is not None:
            st.subheader("📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Claims", len(sample_df))
            with col2:
                fraud_rate = (sample_df['fraud_reported'] == 'Y').mean()
                st.metric("Fraud Rate", f"{fraud_rate:.1%}")
            with col3:
                avg_claim = sample_df['total_claim_amount'].mean()
                st.metric("Avg Claim", f"${avg_claim:,.0f}")
            with col4:
                total_claims_value = sample_df['total_claim_amount'].sum()
                st.metric("Total Claims Value", f"${total_claims_value:,.0f}")
            
            col1, col2 = st.columns(2)
            with col1:
                fraud_by_state = sample_df.groupby('policy_state')['fraud_reported'].apply(lambda x: (x == 'Y').mean()).reset_index()
                fraud_by_state.columns = ['State', 'Fraud_Rate']
                fig_state = px.bar(fraud_by_state, x='State', y='Fraud_Rate', title="Fraud Rate by State")
                st.plotly_chart(fig_state, use_container_width=True)
            with col2:
                fig_age = px.histogram(sample_df, x='age', color='fraud_reported', title="Age Distribution by Fraud Status", nbins=20)
                st.plotly_chart(fig_age, use_container_width=True)
            
            st.subheader("💰 Claim Amount Analysis")
            fig_claim = px.box(sample_df, x='fraud_reported', y='total_claim_amount', title="Claim Amount Distribution by Fraud Status")
            st.plotly_chart(fig_claim, use_container_width=True)
            
            st.subheader("🔗 Feature Correlations")
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
            corr_matrix = sample_df[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, title="Feature Correlation Matrix", aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Sample data not available for analytics dashboard.")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime, timedelta
import re
import json
import warnings
import time
from collections import Counter
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Leads Analytics Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

API_URL = "https://script.google.com/macros/s/AKfycbxRkgVxWW1rpQfUvWDIOG0DI7dh1EawD441cWHFzeLkn8t1u6QtfOmw-NGxgHEZykUU/exec"

# Country code mapping
COUNTRY_CODES = {
    '+1': 'USA/Canada',
    '+91': 'India',
    '+44': 'United Kingdom',
    '+61': 'Australia',
    '+971': 'UAE',
    '+65': 'Singapore',
    '+60': 'Malaysia',
    '+86': 'China',
    '+49': 'Germany',
    '+33': 'France',
    '+81': 'Japan',
    '+82': 'South Korea',
    '+7': 'Russia',
    '+55': 'Brazil',
    '+34': 'Spain',
    '+39': 'Italy',
    '+46': 'Sweden',
    '+47': 'Norway',
    '+358': 'Finland',
    '+31': 'Netherlands',
    '+32': 'Belgium',
    '+41': 'Switzerland',
    '+43': 'Austria',
    '+45': 'Denmark',
    '+48': 'Poland',
    '+420': 'Czech Republic',
    '+27': 'South Africa',
    '+20': 'Egypt',
    '+234': 'Nigeria',
    '+254': 'Kenya',
    '+92': 'Pakistan',
    '+880': 'Bangladesh',
    '+94': 'Sri Lanka',
    '+95': 'Myanmar',
    '+66': 'Thailand',
    '+84': 'Vietnam',
    '+62': 'Indonesia',
    '+63': 'Philippines',
    '+64': 'New Zealand'
}

# ---------------- OPTIMIZED DATA LOADING ----------------
@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    """Optimized data loading from Google Sheets"""
    try:
        start_time = time.time()
        
        # Use session for connection pooling
        session = requests.Session()
        response = session.get(API_URL, timeout=30)
        
        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code}", 0
        
        response_data = response.json()
        
        if not response_data.get('success', False):
            return None, response_data.get('error', 'API Error'), 0
        
        if 'data' not in response_data or not response_data['data']:
            return pd.DataFrame(), "No data found", 0
        
        # Convert to DataFrame efficiently
        df = pd.DataFrame(response_data['data'])
        
        if df.empty:
            return pd.DataFrame(), "Empty data", 0
        
        # Optimize date parsing - FIXED APPROACH
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
        
        for col in date_columns:
            try:
                # First convert to datetime
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                
                # Convert to naive datetime (no timezone) for easier filtering
                if df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_convert(None)  # Convert to local time
                else:
                    df[col] = df[col].dt.tz_localize(None)  # Remove timezone if present
                
            except Exception as e:
                st.sidebar.warning(f"Could not parse {col}: {str(e)[:50]}")
        
        # Clean numeric columns - FIXED REVENUE PARSING
        if 'Total Revenue' in df.columns:
            try:
                # Convert to string first to handle any formatting
                df['Total Revenue'] = df['Total Revenue'].astype(str)
                # Remove currency symbols, commas, and other non-numeric characters
                df['Total Revenue'] = df['Total Revenue'].str.replace(r'[₹$,]', '', regex=True)
                df['Total Revenue'] = pd.to_numeric(df['Total Revenue'], errors='coerce')
            except Exception as e:
                st.sidebar.warning(f"Could not parse Total Revenue: {str(e)[:50]}")
                df['Total Revenue'] = pd.to_numeric(df['Total Revenue'], errors='coerce')
        
        numeric_columns = ['Number of times contacted', 'Number of Sales Activities', 'Days To Close']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        load_time = time.time() - start_time
        return df, f"Loaded {len(df)} records in {load_time:.2f}s", load_time
        
    except requests.exceptions.Timeout:
        return None, "Request timeout", 0
    except Exception as e:
        return None, f"Error: {str(e)}", 0

# ---------------- OPTIMIZED PROCESSING FUNCTIONS ----------------
@st.cache_data
def process_email_validation(df):
    """Fast email validation"""
    if 'Email' not in df.columns:
        return pd.DataFrame()
    
    email_issues = []
    
    for idx, email in df['Email'].items():
        if pd.isna(email) or str(email).strip() == '':
            email_issues.append({
                'Record ID': df.loc[idx, 'Record ID'] if 'Record ID' in df.columns else idx,
                'Email': '',
                'Issue': 'Missing email'
            })
            continue
        
        email_str = str(email).lower().strip()
        
        if '@gmal.com' in email_str:
            email_issues.append({
                'Record ID': df.loc[idx, 'Record ID'] if 'Record ID' in df.columns else idx,
                'Email': email_str,
                'Issue': 'Incorrect domain: gmal.com'
            })
        elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email_str):
            email_issues.append({
                'Record ID': df.loc[idx, 'Record ID'] if 'Record ID' in df.columns else idx,
                'Email': email_str,
                'Issue': 'Invalid email format'
            })
    
    return pd.DataFrame(email_issues)

@st.cache_data
def analyze_phone_numbers_comprehensive(df):
    """Comprehensive phone number analysis with country code detection"""
    if 'Phone Number' not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Create a copy for analysis
    phone_analysis_df = df[['Phone Number']].copy()
    results = []
    
    for idx, phone in phone_analysis_df['Phone Number'].items():
        record_data = {
            'Index': idx,
            'Original': phone,
            'Cleaned': '',
            'Country': 'Unknown',
            'Country_Code': 'Unknown',
            'Is_Valid': False,
            'Length': 0,
            'Has_Plus': False,
            'Format_Type': 'Unknown'
        }
        
        if pd.isna(phone) or phone == '':
            record_data['Format_Type'] = 'Missing/Empty'
            results.append(record_data)
            continue
        
        phone_str = str(phone).strip()
        record_data['Length'] = len(phone_str)
        record_data['Has_Plus'] = phone_str.startswith('+')
        
        # Clean the phone number
        cleaned = re.sub(r'[\s\-\(\)\.]', '', phone_str)
        record_data['Cleaned'] = cleaned
        
        # Detect country code
        country_detected = 'Unknown'
        country_code_detected = 'Unknown'
        
        for code, country in COUNTRY_CODES.items():
            if cleaned.startswith(code.replace('+', '')):
                country_detected = country
                country_code_detected = code
                break
            elif cleaned.startswith(code):
                country_detected = country
                country_code_detected = code
                break
        
        # Check for Indian numbers without +91
        if country_detected == 'Unknown':
            if cleaned.startswith('91') and len(cleaned) >= 12:
                country_detected = 'India'
                country_code_detected = '+91'
            elif cleaned.startswith('0') and (len(cleaned) == 11 or len(cleaned) == 10):
                country_detected = 'India (Local)'
                country_code_detected = '+91'
            elif cleaned.isdigit() and len(cleaned) == 10:
                country_detected = 'India (10 digit)'
                country_code_detected = '+91'
        
        record_data['Country'] = country_detected
        record_data['Country_Code'] = country_code_detected
        
        # Determine format type
        if '+' in phone_str and any(c.isalpha() for c in phone_str):
            record_data['Format_Type'] = 'Contains letters with +'
        elif '+' in phone_str:
            record_data['Format_Type'] = 'International format'
        elif phone_str.startswith('0'):
            record_data['Format_Type'] = 'Local format (starts with 0)'
        elif '(' in phone_str and ')' in phone_str:
            record_data['Format_Type'] = 'Contains parentheses'
        elif '-' in phone_str:
            record_data['Format_Type'] = 'Contains hyphens'
        elif ' ' in phone_str:
            record_data['Format_Type'] = 'Contains spaces'
        elif len(cleaned) < 10:
            record_data['Format_Type'] = 'Too short'
        elif len(cleaned) > 15:
            record_data['Format_Type'] = 'Too long'
        elif cleaned.isdigit():
            record_data['Format_Type'] = 'Digits only'
            record_data['Is_Valid'] = True
        else:
            record_data['Format_Type'] = 'Other format'
        
        results.append(record_data)
    
    # Convert to DataFrame
    analysis_df = pd.DataFrame(results)
    
    # Create summary tables
    # 1. Country distribution
    country_dist = analysis_df['Country'].value_counts().reset_index()
    country_dist.columns = ['Country', 'Count']
    
    # 2. Format type distribution
    format_dist = analysis_df['Format_Type'].value_counts().reset_index()
    format_dist.columns = ['Format_Type', 'Count']
    
    # 3. Country code distribution
    code_dist = analysis_df['Country_Code'].value_counts().reset_index()
    code_dist.columns = ['Country_Code', 'Count']
    
    # 4. Invalid/problematic phones
    invalid_phones = analysis_df[
        (analysis_df['Format_Type'].isin(['Too short', 'Too long', 'Contains letters with +', 'Other format'])) |
        (analysis_df['Country'] == 'Unknown')
    ].copy()
    
    if 'Index' in invalid_phones.columns:
        invalid_phones = invalid_phones.merge(
            df[['Phone Number']].reset_index(),
            left_on='Index',
            right_on='index',
            how='left'
        ).drop('index', axis=1)
    
    return country_dist, format_dist, code_dist, invalid_phones

# ---------------- FIXED DATE FILTERING FUNCTION ----------------
def apply_date_filter_fixed(df, start_date, end_date, date_column='Create Date'):
    """Fixed date filtering that ensures accurate date-only comparison"""
    if date_column not in df.columns:
        return df
    
    if start_date is None or end_date is None:
        return df
    
    try:
        # Create a copy to avoid modifying original
        filtered_df = df.copy()
        
        # Ensure the date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_column]):
            filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors='coerce')
        
        # Remove timezone if present
        if filtered_df[date_column].dt.tz is not None:
            filtered_df[date_column] = filtered_df[date_column].dt.tz_localize(None)
        
        # Extract date part for comparison (ignore time)
        filtered_df['_date_only'] = filtered_df[date_column].dt.date
        
        # Convert input dates to date objects
        start_date_obj = pd.Timestamp(start_date).date()
        end_date_obj = pd.Timestamp(end_date).date()
        
        # Apply filter
        mask = (filtered_df['_date_only'] >= start_date_obj) & (filtered_df['_date_only'] <= end_date_obj)
        result_df = filtered_df[mask].copy()
        
        # Remove helper column
        if '_date_only' in result_df.columns:
            result_df = result_df.drop('_date_only', axis=1)
        
        return result_df
        
    except Exception as e:
        st.sidebar.error(f"Date filter error: {str(e)[:50]}")
        return df

# ---------------- COMPREHENSIVE ANALYSIS FUNCTIONS ----------------
@st.cache_data
def analyze_lead_quality(df):
    """Analyze lead quality metrics"""
    analysis = {}
    
    # 1. Lead Status Distribution
    if 'Lead Status' in df.columns:
        status_dist = df['Lead Status'].value_counts().reset_index()
        status_dist.columns = ['Lead Status', 'Count']
        analysis['status_distribution'] = status_dist
    
    # 2. Course-wise Analysis
    if 'Course' in df.columns:
        course_dist = df['Course'].value_counts().reset_index()
        course_dist.columns = ['Course', 'Count']
        analysis['course_distribution'] = course_dist
    
    # 3. Contact Frequency Analysis
    if 'Number of times contacted' in df.columns:
        # Binned contact frequency
        bins = [0, 1, 3, 5, 10, float('inf')]
        labels = ['0-1', '2-3', '4-5', '6-10', '10+']
        df['Contact_Frequency_Bin'] = pd.cut(
            df['Number of times contacted'], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        )
        contact_bins = df['Contact_Frequency_Bin'].value_counts().reset_index()
        contact_bins.columns = ['Contact Frequency', 'Count']
        analysis['contact_frequency'] = contact_bins
    
    # 4. Days To Close Analysis
    if 'Days To Close' in df.columns:
        # Binned days to close
        bins = [0, 7, 14, 30, 60, float('inf')]
        labels = ['0-7 days', '8-14 days', '15-30 days', '31-60 days', '60+ days']
        df['Days_To_Close_Bin'] = pd.cut(
            df['Days To Close'], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        )
        days_bins = df['Days_To_Close_Bin'].value_counts().reset_index()
        days_bins.columns = ['Days To Close Range', 'Count']
        analysis['days_to_close_bins'] = days_bins
    
    # 5. Revenue Analysis
    if 'Total Revenue' in df.columns:
        try:
            # Revenue by course
            if 'Course' in df.columns:
                revenue_by_course = df.groupby('Course')['Total Revenue'].apply(
                    lambda x: pd.to_numeric(x, errors='coerce').sum()
                ).reset_index()
                revenue_by_course.columns = ['Course', 'Total Revenue']
                revenue_by_course = revenue_by_course.sort_values('Total Revenue', ascending=False)
                analysis['revenue_by_course'] = revenue_by_course
        except:
            pass
    
    # 6. Lead Source Analysis
    if 'Lead Source' in df.columns:
        source_dist = df['Lead Source'].value_counts().reset_index()
        source_dist.columns = ['Lead Source', 'Count']
        analysis['lead_source_distribution'] = source_dist
    
    # 7. Contact Owner Performance
    if 'Contact owner' in df.columns:
        owner_stats = df['Contact owner'].value_counts().reset_index()
        owner_stats.columns = ['Contact Owner', 'Count']
        analysis['owner_performance'] = owner_stats
        
        # Combine with revenue if available
        if 'Total Revenue' in df.columns:
            owner_revenue = df.groupby('Contact owner')['Total Revenue'].apply(
                lambda x: pd.to_numeric(x, errors='coerce').sum()
            ).reset_index()
            owner_revenue.columns = ['Contact Owner', 'Total Revenue']
            analysis['owner_revenue'] = owner_revenue
    
    # 8. Conversion Rate by Source
    if 'Lead Source' in df.columns and 'Lead Status' in df.columns:
        conversion_data = df.groupby(['Lead Source', 'Lead Status']).size().unstack(fill_value=0).reset_index()
        analysis['conversion_by_source'] = conversion_data
    
    return analysis

# ---------------- MAIN APP ----------------
def main():
    st.title("📊 Leads Analytics Dashboard")
    
    # Show loading state
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.info("🔄 Loading data from Google Sheets...")
        
        # Load data
        df, message, load_time = load_data()
    
    # Clear loading placeholder
    loading_placeholder.empty()
    
    if df is None:
        st.error(f"❌ Failed to load data: {message}")
        return
    
    if df.empty:
        st.warning("⚠️ No data available")
        return
    
    # Show success message
    st.success(f"✅ {message}")
    
    # ---------------- ENHANCED DATE FILTER ----------------
    st.sidebar.header("🔍 Filters")
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Date filter options
    st.sidebar.subheader("📅 Date Filter")
    
    # Date selection mode
    date_mode = st.sidebar.radio(
        "Select Date Mode",
        ["Single Date", "Date Range", "All Dates"],
        horizontal=True
    )
    
    start_date = None
    end_date = None
    
    if 'Create Date' in df.columns:
        # Get date range from data
        data_min_date = df['Create Date'].min().date()
        data_max_date = df['Create Date'].max().date()
        
        if date_mode == "Single Date":
            selected_date = st.sidebar.date_input(
                "Select Date",
                value=data_max_date,
                min_value=data_min_date,
                max_value=data_max_date
            )
            start_date = selected_date
            end_date = selected_date
            
        elif date_mode == "Date Range":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=data_min_date,
                    min_value=data_min_date,
                    max_value=data_max_date,
                    key="start_date_input"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=data_max_date,
                    min_value=data_min_date,
                    max_value=data_max_date,
                    key="end_date_input"
                )
            
            if start_date and end_date:
                if start_date > end_date:
                    st.sidebar.warning("Start date cannot be after end date")
                    start_date, end_date = end_date, start_date
        else:  # All Dates
            start_date = data_min_date
            end_date = data_max_date
    
    # Quick date presets
    st.sidebar.subheader("📆 Quick Date Presets")
    
    if 'Create Date' in df.columns and date_mode != "All Dates":
        preset_col1, preset_col2 = st.sidebar.columns(2)
        
        today = datetime.now().date()
        
        with preset_col1:
            if st.button("Today", use_container_width=True):
                start_date = today
                end_date = today
                st.rerun()
            
            if st.button("Yesterday", use_container_width=True):
                yesterday = today - timedelta(days=1)
                start_date = yesterday
                end_date = yesterday
                st.rerun()
        
        with preset_col2:
            if st.button("Last 7 Days", use_container_width=True):
                end_date = today
                start_date = end_date - timedelta(days=6)
                st.rerun()
            
            if st.button("This Month", use_container_width=True):
                start_date = today.replace(day=1)
                end_date = today
                st.rerun()
    
    # Other filters
    st.sidebar.subheader("🎯 Other Filters")
    
    # Course Filter
    if 'Course' in df.columns:
        course_options = ['All Courses'] + sorted(df['Course'].dropna().unique().tolist())
        selected_course = st.sidebar.selectbox("Course", options=course_options)
    else:
        selected_course = 'All Courses'
    
    # Contact Owner Filter
    if 'Contact owner' in df.columns:
        owner_options = ['All Owners'] + sorted(df['Contact owner'].dropna().unique().tolist())
        selected_owner = st.sidebar.selectbox("Contact Owner", options=owner_options)
    else:
        selected_owner = 'All Owners'
    
    # Lead Status Filter
    if 'Lead Status' in df.columns:
        status_options = ['All Statuses'] + sorted(df['Lead Status'].dropna().unique().tolist())
        selected_status = st.sidebar.selectbox("Lead Status", options=status_options)
    else:
        selected_status = 'All Statuses'
    
    # Apply button
    st.sidebar.subheader("⚙️ Apply Filters")
    apply_filters = st.sidebar.button("Apply Filters", type="primary", use_container_width=True)
    
    # Store filter states
    if apply_filters or 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = True
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.selected_course = selected_course
        st.session_state.selected_owner = selected_owner
        st.session_state.selected_status = selected_status
        st.session_state.date_mode = date_mode
    
    # Use stored filter values
    if 'filters_applied' in st.session_state:
        start_date = st.session_state.start_date
        end_date = st.session_state.end_date
        selected_course = st.session_state.selected_course
        selected_owner = st.session_state.selected_owner
        selected_status = st.session_state.selected_status
        date_mode = st.session_state.date_mode
    
    # Apply filters using FIXED date filtering
    filtered_df = df.copy()
    filter_changes = []
    
    # Apply date filter using FIXED function
    if start_date and end_date and 'Create Date' in filtered_df.columns:
        before_count = len(filtered_df)
        filtered_df = apply_date_filter_fixed(filtered_df, start_date, end_date, 'Create Date')
        
        if len(filtered_df) < before_count:
            if start_date == end_date:
                filter_changes.append(f"📅 Date: {start_date}")
            else:
                filter_changes.append(f"📅 Date: {start_date} to {end_date}")
    
    # Apply other filters
    if selected_course != 'All Courses':
        before_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['Course'] == selected_course]
        if len(filtered_df) < before_count:
            filter_changes.append(f"📚 Course: {selected_course}")
    
    if selected_owner != 'All Owners':
        before_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['Contact owner'] == selected_owner]
        if len(filtered_df) < before_count:
            filter_changes.append(f"👤 Owner: {selected_owner}")
    
    if selected_status != 'All Statuses':
        before_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['Lead Status'] == selected_status]
        if len(filtered_df) < before_count:
            filter_changes.append(f"📊 Status: {selected_status}")
    
    # Show active filters
    if filter_changes:
        st.sidebar.success("**Active Filters:**\n" + "\n".join([f"• {f}" for f in filter_changes]))
    
    # Filter summary
    st.sidebar.divider()
    st.sidebar.metric("Total Records", f"{len(df):,}")
    st.sidebar.metric("Filtered Records", f"{len(filtered_df):,}")
    
    # Clear filters button
    if st.sidebar.button("🗑️ Clear All Filters", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No records match the selected filters.")
        return
    
    # ---------------- MAIN KPIs ----------------
    st.subheader("📈 Key Performance Indicators")
    
    # Create metrics row with 3 KPIs
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    
    with kpi_col1:
        # Total Leads
        total_leads = len(filtered_df)
        delta_leads = total_leads - len(df) if total_leads != len(df) else None
        delta_label = f"{delta_leads:,}" if delta_leads else None
        st.metric("Total Leads", f"{total_leads:,}", delta=delta_label)
    
    with kpi_col2:
        # Total Revenue
        if 'Total Revenue' in filtered_df.columns:
            try:
                # Get numeric revenue
                revenue_series = pd.to_numeric(filtered_df['Total Revenue'], errors='coerce')
                total_revenue = revenue_series.sum()
                
                # Format display
                if pd.isna(total_revenue):
                    st.metric("Total Revenue", "N/A")
                else:
                    if total_revenue >= 10000000:  # 1 crore and above
                        in_crore = total_revenue / 10000000
                        st.metric("Total Revenue", f"₹{in_crore:.2f} Cr")
                    elif total_revenue >= 100000:  # 1 lakh and above
                        in_lakhs = total_revenue / 100000
                        st.metric("Total Revenue", f"₹{in_lakhs:.2f} L")
                    else:
                        st.metric("Total Revenue", f"₹{total_revenue:,.0f}")
            except Exception as e:
                st.metric("Total Revenue", "Error")
        else:
            st.metric("Total Revenue", "N/A")
    
    with kpi_col3:
        # Total Owners
        if 'Contact owner' in filtered_df.columns:
            total_owners = filtered_df['Contact owner'].nunique()
            st.metric("Total Owners", total_owners)
        else:
            st.metric("Total Owners", "N/A")
    
    # ---------------- COMPREHENSIVE ANALYSIS SECTION WITH DOWNLOAD OPTIONS ----------------
    st.subheader("🔍 Detailed Lead Analysis")
    
    # Run comprehensive analysis
    lead_analysis = analyze_lead_quality(filtered_df)
    
    # Create tabs for different analysis sections
    analysis_tabs = st.tabs([
        "📊 Lead Status", 
        "📚 Courses", 
        "📞 Contact Frequency",
        "⏳ Days to Close",
        "💰 Revenue",
        "🌐 Lead Sources",
        "👥 Contact Owners"
    ])
    
    # Tab 1: Lead Status Analysis
    with analysis_tabs[0]:
        if 'status_distribution' in lead_analysis:
            status_data = lead_analysis['status_distribution']
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Lead Status Distribution")
                st.dataframe(status_data, use_container_width=True, height=400)
            
            with col2:
                # Download button
                csv_status = status_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Lead Status",
                    csv_status,
                    "lead_status_distribution.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Chart
                if len(status_data) > 0:
                    fig = px.pie(
                        status_data.head(10),
                        values='Count',
                        names='Lead Status',
                        title="Top 10 Lead Statuses",
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Course Analysis
    with analysis_tabs[1]:
        if 'course_distribution' in lead_analysis:
            course_data = lead_analysis['course_distribution']
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Course Distribution")
                st.dataframe(course_data, use_container_width=True, height=400)
            
            with col2:
                # Download button
                csv_course = course_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Course Data",
                    csv_course,
                    "course_distribution.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Chart
                if len(course_data) > 0:
                    fig = px.bar(
                        course_data.head(10),
                        x='Course',
                        y='Count',
                        title="Top 10 Courses",
                        color='Count'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Contact Frequency Analysis
    with analysis_tabs[2]:
        if 'contact_frequency' in lead_analysis:
            freq_data = lead_analysis['contact_frequency']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Contact Frequency Distribution")
                st.dataframe(freq_data, use_container_width=True, height=400)
                
                # Download frequency data
                csv_freq = freq_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Frequency Distribution",
                    csv_freq,
                    "contact_frequency_distribution.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Chart
                if len(freq_data) > 0:
                    fig = px.pie(
                        freq_data,
                        values='Count',
                        names='Contact Frequency',
                        title="Contact Frequency Distribution",
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Days to Close Analysis
    with analysis_tabs[3]:
        if 'days_to_close_bins' in lead_analysis:
            days_bins = lead_analysis['days_to_close_bins']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Days to Close Distribution")
                st.dataframe(days_bins, use_container_width=True, height=400)
                
                # Download bins data
                csv_days_bins = days_bins.to_csv(index=False)
                st.download_button(
                    "📥 Download Days Distribution",
                    csv_days_bins,
                    "days_to_close_distribution.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Chart
                if len(days_bins) > 0:
                    fig = px.bar(
                        days_bins,
                        x='Days To Close Range',
                        y='Count',
                        title="Days to Close Distribution",
                        color='Count'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Revenue Analysis
    with analysis_tabs[4]:
        # Revenue by Course
        if 'revenue_by_course' in lead_analysis:
            revenue_by_course = lead_analysis['revenue_by_course']
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Revenue by Course")
                st.dataframe(revenue_by_course, use_container_width=True, height=400)
            
            with col2:
                # Download revenue by course
                csv_rev_course = revenue_by_course.to_csv(index=False)
                st.download_button(
                    "📥 Download Revenue by Course",
                    csv_rev_course,
                    "revenue_by_course.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Chart
                if len(revenue_by_course) > 0:
                    fig = px.bar(
                        revenue_by_course.head(10),
                        x='Course',
                        y='Total Revenue',
                        title="Top 10 Courses by Revenue",
                        color='Total Revenue'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: Lead Source Analysis
    with analysis_tabs[5]:
        if 'lead_source_distribution' in lead_analysis:
            source_data = lead_analysis['lead_source_distribution']
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Lead Source Distribution")
                st.dataframe(source_data, use_container_width=True, height=400)
            
            with col2:
                # Download lead source data
                csv_source = source_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Lead Sources",
                    csv_source,
                    "lead_source_distribution.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Chart
                if len(source_data) > 0:
                    fig = px.pie(
                        source_data.head(10),
                        values='Count',
                        names='Lead Source',
                        title="Top 10 Lead Sources",
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Conversion by Source
            if 'conversion_by_source' in lead_analysis:
                conversion_data = lead_analysis['conversion_by_source']
                
                st.markdown("### Conversion by Lead Source")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(conversion_data, use_container_width=True, height=300)
                
                with col2:
                    # Download conversion data
                    csv_conv = conversion_data.to_csv(index=False)
                    st.download_button(
                        "📥 Download Conversion Data",
                        csv_conv,
                        "conversion_by_source.csv",
                        "text/csv",
                        use_container_width=True
                    )
    
    # Tab 7: Contact Owner Analysis
    with analysis_tabs[6]:
        if 'owner_performance' in lead_analysis:
            owner_data = lead_analysis['owner_performance']
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Contact Owner Performance")
                st.dataframe(owner_data, use_container_width=True, height=400)
            
            with col2:
                # Download owner performance data
                csv_owner = owner_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Owner Performance",
                    csv_owner,
                    "contact_owner_performance.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Chart
                if len(owner_data) > 0:
                    fig = px.bar(
                        owner_data.head(10),
                        x='Contact Owner',
                        y='Count',
                        title="Top 10 Contact Owners",
                        color='Count'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Owner Revenue Analysis
            if 'owner_revenue' in lead_analysis:
                owner_revenue_data = lead_analysis['owner_revenue']
                
                st.markdown("### Revenue by Contact Owner")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(owner_revenue_data, use_container_width=True, height=300)
                
                with col2:
                    # Download owner revenue data
                    csv_owner_rev = owner_revenue_data.to_csv(index=False)
                    st.download_button(
                        "📥 Download Owner Revenue",
                        csv_owner_rev,
                        "owner_revenue_analysis.csv",
                        "text/csv",
                        use_container_width=True
                    )
    
    # ---------------- COMPREHENSIVE PHONE NUMBER ANALYSIS ----------------
    if 'Phone Number' in filtered_df.columns:
        st.subheader("🌍 Phone Number Analysis by Country")
        
        # Analyze phone numbers
        country_dist, format_dist, code_dist, invalid_phones = analyze_phone_numbers_comprehensive(filtered_df)
        
        # Create attractive layout with tabs
        phone_tabs = st.tabs(["📊 Country Distribution", "📱 Format Analysis", "🔢 Country Codes", "⚠️ Problematic Numbers"])
        
        with phone_tabs[0]:
            # Country Distribution
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### Country Distribution")
                st.dataframe(
                    country_dist,
                    column_config={
                        "Country": st.column_config.TextColumn("Country", width="medium"),
                        "Count": st.column_config.NumberColumn("Count", format="%d", width="small")
                    },
                    use_container_width=True,
                    height=400
                )
                
                # Download country data
                csv_country = country_dist.to_csv(index=False)
                st.download_button(
                    "📥 Download Country Data",
                    csv_country,
                    "phone_countries.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                if len(country_dist) > 0:
                    # Show top countries in a nice chart
                    top_countries = country_dist.head(10)
                    fig1 = px.pie(
                        top_countries,
                        names='Country',
                        values='Count',
                        title="Top 10 Countries Distribution",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig1.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Quick stats
                    total_phones = country_dist['Count'].sum()
                    india_count = country_dist[country_dist['Country'].str.contains('India')]['Count'].sum()
                    international_count = total_phones - india_count if not pd.isna(india_count) else total_phones
                    
                    st.metric("Total Phone Numbers", f"{total_phones:,}")
                    st.metric("Indian Numbers", f"{india_count:,}")
                    st.metric("International", f"{international_count:,}")
        
        with phone_tabs[1]:
            # Format Analysis
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### Phone Number Formats")
                st.dataframe(
                    format_dist,
                    column_config={
                        "Format_Type": st.column_config.TextColumn("Format Type", width="medium"),
                        "Count": st.column_config.NumberColumn("Count", format="%d", width="small")
                    },
                    use_container_width=True,
                    height=400
                )
                
                # Download format data
                csv_format = format_dist.to_csv(index=False)
                st.download_button(
                    "📥 Download Format Data",
                    csv_format,
                    "phone_formats.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                if len(format_dist) > 0:
                    # Show format distribution
                    top_formats = format_dist.head(10)
                    fig2 = px.bar(
                        top_formats,
                        x='Format_Type',
                        y='Count',
                        title="Top 10 Phone Formats",
                        color='Count',
                        color_continuous_scale='Viridis'
                    )
                    fig2.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)
        
        with phone_tabs[2]:
            # Country Codes
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### Country Code Distribution")
                st.dataframe(
                    code_dist,
                    column_config={
                        "Country_Code": st.column_config.TextColumn("Country Code", width="small"),
                        "Count": st.column_config.NumberColumn("Count", format="%d", width="small")
                    },
                    use_container_width=True,
                    height=400
                )
                
                # Download country code data
                csv_code = code_dist.to_csv(index=False)
                st.download_button(
                    "📥 Download Country Codes",
                    csv_code,
                    "country_codes_distribution.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Show country code reference
                with st.expander("🌐 Country Code Reference"):
                    ref_df = pd.DataFrame(list(COUNTRY_CODES.items()), columns=['Code', 'Country'])
                    st.dataframe(ref_df, use_container_width=True, height=300)
            
            with col2:
                if len(code_dist) > 0:
                    # Show country code distribution
                    fig3 = px.bar(
                        code_dist.head(15),
                        x='Country_Code',
                        y='Count',
                        title="Top 15 Country Codes",
                        color='Count',
                        color_continuous_scale='Plasma'
                    )
                    fig3.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig3, use_container_width=True)
        
        with phone_tabs[3]:
            # Problematic Numbers
            if len(invalid_phones) > 0:
                st.warning(f"⚠️ Found {len(invalid_phones)} potentially problematic phone numbers")
                
                # Show problematic numbers
                st.dataframe(
                    invalid_phones[['Original', 'Country', 'Format_Type', 'Length']].head(20),
                    column_config={
                        "Original": st.column_config.TextColumn("Phone Number", width="medium"),
                        "Country": st.column_config.TextColumn("Detected Country", width="medium"),
                        "Format_Type": st.column_config.TextColumn("Issue", width="medium"),
                        "Length": st.column_config.NumberColumn("Length", format="%d", width="small")
                    },
                    use_container_width=True,
                    height=300
                )
                
                # Download problematic phones
                if 'Original' in invalid_phones.columns:
                    csv_problematic = invalid_phones[['Original', 'Country', 'Format_Type']].to_csv(index=False)
                    st.download_button(
                        "📥 Download Problematic Numbers",
                        csv_problematic,
                        "problematic_phones.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.success("✅ No problematic phone numbers found!")
    
    # ---------------- REASON ANALYSIS ----------------
    st.subheader("🎯 Prospect Reasons Analysis")
    
    reason_columns = [
        'Future Prospect Reasons',
        'Hot Prospect Reason', 
        'Neutral Prospect Reasons',
        'Not Connected Reasons',
        'Not Interested Reasons',
        'Other Enquiry Reasons',
        'Prospect Reasons'
    ]
    
    available_reasons = [col for col in reason_columns if col in filtered_df.columns]
    
    if available_reasons:
        tabs = st.tabs([col[:12] + "..." for col in available_reasons[:5]])  # Limit to 5 tabs
        
        for i, reason_col in enumerate(available_reasons[:5]):
            with tabs[i]:
                reasons = filtered_df[reason_col].dropna()
                
                if not reasons.empty:
                    # Fast value counts
                    reason_counts = reasons.value_counts().reset_index()
                    reason_counts.columns = ['Reason', 'Count']
                    
                    # Display efficiently
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.dataframe(
                            reason_counts,
                            use_container_width=True,
                            height=300
                        )
                    
                    with col_b:
                        # Download
                        csv = reason_counts.to_csv(index=False)
                        st.download_button(
                            "📥 Download",
                            csv,
                            f"{reason_col.replace(' ', '_')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    # Simple chart
                    if len(reason_counts) > 0:
                        top_10 = reason_counts.head(10)
                        fig = px.bar(
                            top_10,
                            x='Reason',
                            y='Count',
                            title=f"Top 10 - {reason_col[:20]}"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No data in {reason_col}")
    else:
        st.info("No reason columns found")
    
    # ---------------- EMAIL VALIDATION ----------------
    st.subheader("📧 Email Validation")
    
    email_issues = process_email_validation(filtered_df)
    
    if not email_issues.empty:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.warning(f"Found {len(email_issues)} email issues")
            st.dataframe(email_issues, use_container_width=True, height=250)
        
        with col2:
            st.metric("Total Issues", len(email_issues))
            
            gmal_count = (email_issues['Issue'] == 'Incorrect domain: gmal.com').sum()
            st.metric("gmal.com issues", gmal_count)
            
            # Download
            csv = email_issues.to_csv(index=False)
            st.download_button(
                "📥 Download Email Issues",
                csv,
                "email_issues.csv",
                "text/csv",
                use_container_width=True
            )
    else:
        st.success("✅ All emails appear valid!")
    
    # ---------------- DATA EXPORT ----------------
    st.subheader("📥 Export Options")
    
    export_cols = st.columns(4)
    
    with export_cols[0]:
        # CSV Export
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            "📄 Download CSV",
            csv_data,
            f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with export_cols[1]:
        # Excel Export
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, sheet_name='Leads_Data', index=False)
            
        st.download_button(
            "📊 Download Excel",
            output.getvalue(),
            f"leads_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with export_cols[2]:
        # View data button
        if st.button("👁️ View Filtered Data", use_container_width=True):
            st.dataframe(filtered_df, use_container_width=True, height=400)
    
    with export_cols[3]:
        # Quick refresh button
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()
    
    # Footer
    st.divider()
    st.caption(f"Dashboard loaded in {load_time:.2f}s | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

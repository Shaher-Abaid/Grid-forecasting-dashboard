import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.toggle_switch import st_toggle_switch
import warnings
warnings.filterwarnings('ignore')

# ---------------------- DARK MODE FIX FOR METRICS ---------------------- #
def inject_metric_card_css():
    st.markdown("""
        <style>
        div[data-testid="metric-container"] {
            background-color: #111 !important;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 10px;
            color: #f0f0f0 !important;
        }
        div[data-testid="metric-container"] > label {
            color: #f0f0f0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ---------------------- PAGE SETUP ---------------------- #
st.set_page_config(
    page_title="EDA & ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- DATA LOADING ---------------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("Model.csv", parse_dates=["Start_Time"])
    return df

df = load_data()

# ---------------------- ADVANCED FILTER ---------------------- #
def apply_filters(df):
    st.sidebar.subheader("🔍 Advanced Filters")
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != "Start_Time"]
    filtered_df = df.copy()

    for col in numeric_cols:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        selected_range = st.sidebar.slider(
            f"{col.replace('_', ' ').title()}",
            min_val, max_val, (min_val, max_val),
            key=f"filter_{col}"
        )
        filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]

    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        values = df[col].dropna().unique().tolist()
        selected_values = st.sidebar.multiselect(f"{col}:", values, default=values)
        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    return filtered_df

# ---------------------- SIDEBAR NAVIGATION ---------------------- #
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["🏠 Home", "📈 Load Trends & Renewable Insights" ,"📊 Univariate", "🔗 Bivariate", "🌐 Multivariate", "⏳ Time Series Analysis", "🤖 ML Predictions"],
        icons=["house", "bar-chart", "scatter-chart", "layers", "clock-history", "robot"],
        menu_icon="cast",
        default_index=0
    )

# ---------------------- ADVANCED FILTERING ---------------------- #
if st.sidebar.checkbox("🔧 Enable Advanced Filtering", value=False):
    df_filtered = apply_filters(df)
else:
    df_filtered = df

# ---------------------- HOME ---------------------- #
if selected == "🏠 Home":
    # --- Title and Subtitle ---
    st.markdown('<h1 style="text-align:center; color:#1f77b4;">⚡ Grid Forecasting Project Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align:center; color:gray;">Machine Learning meets Energy Forecasting – Germany’s Grid Unveiled</h3>', unsafe_allow_html=True)

    # --- Project Overview ---
    colored_header("⚙️ Project Overview", "National-Scale Forecasting of Electricity Load and Renewable Generation inGermany Using Weather and Time Features", "blue-70")

    st.markdown("""
    This interactive dashboard brings together:

    🔹 **Model 1 – Grid Load Forecasting**  
    A regression model predicting national electricity demand using weather, time, and generation patterns.

    🔹 **Model 2 – Renewable Share Classification**  
    A classifier that forecasts whether renewables will supply **≥ 50%** of grid consumption in a given hour.

    📡 Powered by historical data from Germany's energy grid, enriched with weather features and engineered insights.
    """)

    # --- Styling for Metric Cards ---#
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 15px;
        border-radius: 10px;
    }
    label[data-testid="stMetricLabel"] > div,
    div[data-testid="stMetricValue"] > div {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

    #KPIS
    total_hours = len(df_filtered)
    start_date = df_filtered['Start_Time'].min().strftime('%Y-%m-%d')
    end_date = df_filtered['Start_Time'].max().strftime('%Y-%m-%d')
    avg_demand = df_filtered['Grid Load (Consumption)'].mean()
    renewable_dominance_pct = (df_filtered['Renewable_vs_Consumption_Ratio'] >= 0.5).mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Time Range", f"2015 → 2024")
    col2.metric("⚡ Avg Grid Load", f"{avg_demand:,.0f} MW")
    col3.metric("🌿 Renewable > 50%", f"{renewable_dominance_pct:.1f}% of hours")
    col4.metric("📈 Total Records", f"{total_hours:,}")

    # --- Strategic Insights 
    st.subheader("🔍 Project Insights")
    st.markdown("""
    ### 🔋 Grid Load Drivers
    - **Total Grid Load incl. Hydro (0.99)** and **Actual Generation (0.89)** are the most positively correlated.
    - **Temperature (0.42)** and **Residual Load (0.54)** are key demand indicators.
    - Conventional sources like **Hard Coal (–0.55)** and **Lignite (–0.36)** show strong inverse correlation.

    ### 🌞 Renewable Share Patterns
    - The **Renewable_vs_Consumption_Ratio** is bimodal: <0.4 during low generation or high demand, >0.6 when renewables dominate.
    - **Lowest shares**: August–September (~22%); **Highest**: March–May (~35%).

    ### 🧭 Time-Based Dynamics
    - **Weekends** feature lower industrial demand, allowing renewables to more easily exceed 50%.
    - **Daily load** peaks 11:00–18:00, dips 03:00–05:00.
    - Outliers often align with holidays or weekends.

    ### ⚙️ Load Level vs Energy Mix
    - As demand increases, reliance on **dispatchable sources** like Lignite and Gas rises.
    - **Renewables’ proportional share drops** at high load levels, despite total output.
    - **Nuclear** remains a constant baseload source.

    ### 🌡️ Temperature Effects
    - Cold winters cause significant demand surges.
    - Summer impact is weaker, likely due to lower A/C penetration.

    ### 📌 Summary Insight
    - Renewable dominance is not purely generation-driven—it’s a **balance between production and consumption**.
    - **Low demand + high renewable output** = optimal condition for >50% share.
    """)

    # --- Data Sources ---#
    st.subheader("📚 Primary Data Sources")
    st.markdown("""
    **SMARD (Strommarktdaten)**  
    Germany’s official electricity market platform.  
    ↪ Federal Network Agency for Electricity, Gas, Telecommunications, Post and Railway  
    ↪ Real-time + historical generation & consumption  
    🔗 [https://www.smard.de/](https://www.smard.de/)

    **Meteostat**  
    Historical global weather data from official meteorological stations.  
    ↪ Hourly records, Python API support  
    🔗 [https://meteostat.net/](https://meteostat.net/)
    """)
    
        #--------------- 📈 Load Trends & Renewable Insights---------------------------------#

elif selected == "📈 Load Trends & Renewable Insights":
    st.header("📈 Load Trends & Renewable Insights")

    # datetime conversion
    df['Date'] = pd.to_datetime(df['Date'])

    with st.expander("1️⃣ How does electricity consumption vary over time?"):
        fig = px.line(df, x='Date', y='Grid Load (Consumption)',
                      title='Electricity Consumption Over Time',
                      labels={'Grid Load (Consumption)': 'MW'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("- 📈 Consumption peaks in winter months (heating demand) and dips slightly in summer.")

    with st.expander("2️⃣ How does temperature affect consumption?"):
        sampled_df = df.sample(500)
        fig2 = px.scatter(sampled_df,
                         y='Temperature (°C)',
                         x='Grid Load (Consumption)',
                         color='Month',
                         title='Effect of Temperature on Electricity Consumption',
                         labels={'Grid Load (Consumption)': 'Consumption (MW)',
                                 'Temperature (°C)': 'Temperature (°C)'})
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("- ❄️ Lower temperatures correlate with higher consumption. Winter heating is a key driver.")

    with st.expander("3️⃣ How does load behave on weekends vs weekdays?"):
        weekend_avg = df.groupby('is_weekend')['Grid Load (Consumption)'].mean().reset_index()
        fig3 = px.bar(weekend_avg,
                     x='is_weekend',
                     y='Grid Load (Consumption)',
                     color="is_weekend",
                     title='Average Electricity Consumption: Weekdays vs Weekends',
                     height=500, width=500,
                     labels={'is_weekend': 'Weekend', 'Grid Load (Consumption)': 'Avg Load (MW)'})
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("- 🏭 Weekday demand is higher due to industrial activity. Weekends show ~10k MW lower usage.")

    with st.expander("4️⃣ How do energy sources relate to load levels?"):
        gen_cols = ['Biomass Generation', 'Hydropower Generation', 'Wind Offshore Generation',
                    'Wind Onshore Generation', 'Solar PV Generation', 'Other Renewable Generation',
                    'Nuclear Generation', 'Lignite Generation', 'Hard Coal Generation',
                    'Fossil Gas Generation', 'Pumped Storage Supply', 'Other Conventional Generation']
        renw_cols = ['Biomass Generation', 'Hydropower Generation', 'Wind Offshore Generation',
                     'Wind Onshore Generation', 'Solar PV Generation', 'Other Renewable Generation']

        df_load = pd.concat([
            pd.qcut(df['Grid Load (Consumption)'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High']).rename('Load Category'),
            df[gen_cols]
        ], axis=1)

        df_load_renw = pd.concat([
            pd.qcut(df['Grid Load (Consumption)'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High']).rename('Load Category'),
            df[renw_cols]
        ], axis=1)

        Categorized_df = df_load.groupby('Load Category')[gen_cols].mean().reset_index()
        Categorized_df_renw = df_load_renw.groupby('Load Category')[renw_cols].mean().reset_index()

        fig4 = px.bar(Categorized_df, x='Load Category', y=gen_cols, barmode='group',
                      title='Generation Source Output by Load Level', height=600)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("- 🔼 Lignite, Coal, and Gas ramp up as load increases. Nuclear stays flat. Renewables contribute most under low/mid load.")

        fig4_1 = px.bar(Categorized_df_renw, x='Load Category', y=renw_cols, barmode='group',
                        title='Renewable Generation by Load Level', height=600)
        st.plotly_chart(fig4_1, use_container_width=True)
        st.markdown("- 🌿 Renewable energy output doesn't scale linearly with demand — limited by resource availability.")

    with st.expander("5️⃣ How frequently does renewable energy cover >50% of demand?"):
        fig6 = px.pie(df,
                      names=(df["Renewable_vs_Consumption_Ratio"] > 0.5).map({True: "Over 50%", False: "Under 50%"}),
                      title='Renewable Energy Coverage Ratio')
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("- 🔋 Renewables exceed 50% of demand in a significant number of hours, especially under low-load conditions.")

    with st.expander("6️⃣ What features drive high renewable share?"):
        corr_df2 = df[[
            'Temperature (°C)', 'Dew Point (°C)', 'Relative Humidity (%)',
            'Precipitation (mm)', 'Snow Depth (mm)', 'Wind Direction (°)',
            'Wind Speed (m/s)', 'Wind Gust (m/s)', 'Pressure (hPa)',
            'Sunshine Duration (min)', 'Weather Code', 'is_weekend',
            'Biomass Generation', 'Hydropower Generation', 'Wind Offshore Generation',
            'Wind Onshore Generation', 'Solar PV Generation', 'Other Renewable Generation',
            'Grid Load (Consumption)', 'Total Grid Load incl. Hydro', 'Residual Load', 'High_Renewable_Share']
        ].corr()['High_Renewable_Share'].drop('High_Renewable_Share').sort_values(ascending=True).round(4) * 100
        corr_df2 = corr_df2.reset_index()
        corr_df2.columns = ["Feature", "Correlation (%)"]

        fig6_1 = px.bar(
            corr_df2,
            x="Correlation (%)",
            y="Feature",
            orientation='h', height=600,
            title="Correlation with High Renewable Share",
            labels={"Correlation (%)": "Correlation (%)", "Feature": "Features"},
            color="Correlation (%)",
            color_continuous_scale="Tealrose")
        st.plotly_chart(fig6_1, use_container_width=True)
        st.markdown("- 📊 Top drivers include wind speed, wind onshore/offshore, solar PV, and sunshine duration.")

    with st.expander("7️⃣ How does renewable generation compare to total consumption?"):
        fig8 = px.scatter(df,
                          x='Grid Load (Consumption)',
                          y='Total_renewable_generation',
                          color='High_Renewable_Share',
                          symbol='High_Renewable_Share',
                          title='Renewable Generation vs Load by Class',
                          labels={
                              'Grid Load (Consumption)': 'Grid Load (MW)',
                              'Total_renewable_generation': 'Renewable Generation (MW)',
                              'High_Renewable_Share': 'Renewable Share > 50%'
                          }, height=500)
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("- ✅ Renewable dominance occurs when generation is high and demand is moderate.")

    with st.expander("8️⃣ Which weather conditions are linked to high renewables?"):
        Weather = df[[
            'Temperature (°C)', 'Dew Point (°C)', 'Relative Humidity (%)',
            'Precipitation (mm)', 'Snow Depth (mm)', 'Wind Direction (°)',
            'Wind Speed (m/s)', 'Wind Gust (m/s)', 'Pressure (hPa)',
            'Sunshine Duration (min)', 'Weather Code', 'Total_renewable_generation']]

        Weather_corr = Weather.corr()['Total_renewable_generation'].drop("Total_renewable_generation").sort_values(ascending=False).round(4) * 100
        weather_corr_df = Weather_corr.reset_index()
        weather_corr_df.columns = ['Feature', 'Correlation (%)']
        weather_corr_df['Size'] = weather_corr_df['Correlation (%)'].abs()

        fig7 = px.scatter(
            weather_corr_df,
            x='Feature',
            y='Correlation (%)',
            size='Size',
            color='Correlation (%)',
            color_continuous_scale='Deep', height=500, width=800,
            title='Bubble Chart: Correlation of Weather Features with Renewable Generation')
        fig7.add_hline(y=0, line_dash='dot', line_color='black',
                       annotation_text='Zero Correlation', annotation_position='top right')
        fig7.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("- 🌤️ Wind speed, gusts, and sunshine duration show strong positive correlation with renewable output. Humidity and pressure show negative impact.")

    with st.expander("9️⃣ What is the distribution of Renewable_vs_Consumption_Ratio?"):
        fig10 = px.violin(df, y="Renewable_vs_Consumption_Ratio", x="High_Renewable_Share",
                          animation_frame="Month", color="Weather Group", box=True,
                          labels={"Renewable_vs_Consumption_Ratio": "Renewable Share",
                                  "High_Renewable_Share": "Dominated by Renewables"}, height=600)
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown("- 🧠 Bimodal distribution: One peak under 0.4, another above 0.6. Influenced by weather and seasonal patterns.")







            # ---------------------- UNIVARIATE ---------------------- #
elif selected == "📊 Univariate":
    st.header("📊 Univariate Analysis")
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_filtered.select_dtypes(include='object').columns.tolist()

    tab1, tab2 = st.tabs(["Numerical", "Categorical"])

    with tab1:
        selected_num = st.selectbox("Choose numerical column", numeric_cols)
        fig = px.histogram(df_filtered, x=selected_num, nbins=40, title=f"Distribution of {selected_num}")
        st.plotly_chart(fig, use_container_width=True)

        # Summary for numerical column
        desc = df_filtered[selected_num].describe()
        skewness = df_filtered[selected_num].skew()
        st.markdown(f"""
        #### 📌 Summary of `{selected_num}`
        - **Mean**: `{desc['mean']:.2f}`
        - **Median**: `{desc['50%']:.2f}`
        - **Std Dev**: `{desc['std']:.2f}`
        - **Min / Max**: `{desc['min']:.2f}` / `{desc['max']:.2f}`
        - **Skewness**: `{skewness:.2f}` → {"Right-skewed" if skewness > 0.5 else "Left-skewed" if skewness < -0.5 else "Fairly symmetric"}
        """)

    with tab2:
        selected_cat = st.selectbox("Choose categorical column", categorical_cols[3:])
        cat_counts = df_filtered[selected_cat].value_counts().reset_index()
        cat_counts.columns = ['Category', 'count']
        fig = px.bar(cat_counts, x='Category', y='count', title=f"Counts of {selected_cat}")
        st.plotly_chart(fig, use_container_width=True)

        # Summary for categorical column
        most_common = cat_counts.iloc[0]
        st.markdown(f"""
        #### 📌 Summary of `{selected_cat}`
        - **Unique Categories**: `{df_filtered[selected_cat].nunique()}`
        - **Most Frequent**: `{most_common['Category']}` with `{most_common['count']}` records
        - **Top Categories Share**: `{(most_common['count'] / len(df_filtered) * 100):.1f}%` of total
        """)


                                # ---------------------- BIVARIATE ---------------------- #
elif selected == "🔗 Bivariate":
    st.header("🔗 Bivariate Analysis")
    num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        x = st.selectbox("X Axis", num_cols)
    with col2:
        y = st.selectbox("Y Axis", num_cols, index=1)

    fig = px.scatter(df_filtered, x=x, y=y, trendline="ols", title=f"{x} vs {y}")
    st.plotly_chart(fig, use_container_width=True)

    corr = df_filtered[x].corr(df_filtered[y])
    st.info(f"Pearson Correlation = **{corr:.2f}**")

    if "Grid Load incl. Hydro" in [x, y]:
        st.warning("\u26a0\ufe0f 'Total Grid Load incl. Hydro' includes part of the target load — this may cause misleading correlation!")

                            # ---------------------- MULTIVARIATE ---------------------- #
elif selected == "🌐 Multivariate":
    st.header("🌐 Multivariate Analysis")
    st.caption("Explore relationships between multiple variables and how they interact to influence energy demand and renewable share.")

    # Targeted correlation heatmap (top 20 only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr().abs()

    # Focus: show top 20 variables most correlated with Grid Load
    target_corr = corr_matrix["Grid Load (Consumption)"].drop("Grid Load (Consumption)").sort_values(ascending=False).head(20)
    top_corr_features = target_corr.index.tolist() + ["Grid Load (Consumption)"]
    filtered_corr = df[top_corr_features].corr()

    fig = px.imshow(filtered_corr, text_auto=True, aspect="auto",
                    title="🔍 Top Correlated Features with Grid Load",
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("- 🎯 Focused on the strongest 20 correlations with **Grid Load (Consumption)**.")
    st.markdown("- Helps identify which features influence demand most across multiple dimensions.")

    # Optional 3D plot example
    st.subheader("📊 Interaction Effect: Wind & Solar on Renewable Share")
    fig_3d = px.scatter_3d(df.sample(500),
                           x='Wind Speed (m/s)',
                           y='Solar PV Generation',
                           z='Renewable_vs_Consumption_Ratio',
                           color='High_Renewable_Share',
                           title='3D Interaction: Wind + Solar vs Renewable Share')
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown("- ☀️🌬️ High solar and wind together lead to **higher renewable share**.")

    # Class correlation (with High_Renewable_Share)
    st.subheader("📈 Feature Correlation with Renewable Dominance")
    df['High_Renewable_Share'] = df['High_Renewable_Share'].astype(int)
    numeric_df = df.select_dtypes(include='number')
    class_corr = numeric_df.corr()['High_Renewable_Share'].drop('High_Renewable_Share') * 100
    class_corr = class_corr.sort_values(ascending=True).reset_index()
    class_corr.columns = ['Feature', 'Correlation (%)']

    fig_bar = px.bar(class_corr.tail(10),  # Show top 10 positive correlations only
                     x='Correlation (%)', y='Feature', orientation='h',
                     title='Top Features Correlated with High Renewable Share',
                     color='Correlation (%)',
                     color_continuous_scale='Tealrose')
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("- ✅ Focused on **top 10 features** most associated with high renewable share hours.")
    st.markdown("- 🧠 Although correlations are modest, features like wind speed and solar PV are clearly influential.")



                        # ---------------------- TIME SERIES ---------------------- #
elif selected == "⏳ Time Series Analysis":
    st.header("⏳ Time Series: Renewable Share Patterns")
    st.caption("Explore how the dominance of renewable energy varies across time — by hour, weekday, month, and weekends.")

    # Ensure datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # By hour
    hourly = df.groupby('Start_Time')['High_Renewable_Share'].mean().reset_index()
    fig9_1 = px.line(hourly, x='Start_Time', y='High_Renewable_Share',
                    title='🔄 Renewable Dominance by Hour of Day',
                    labels={'High_Renewable_Share': 'Share > 50%'})
    st.plotly_chart(fig9_1, use_container_width=True)
    st.markdown("- 🕒 Renewable dominance varies throughout the day. Expect lower shares during peak consumption hours.")

    # By day of week
    daily = df.groupby(df['Date'].dt.day_name())['High_Renewable_Share'].mean().reset_index()
    fig9_2 = px.bar(daily, x='Date', y='High_Renewable_Share',
                   title='📅 Renewable Dominance by Day of Week',
                   labels={'High_Renewable_Share': 'Share > 50%'})
    st.plotly_chart(fig9_2, use_container_width=True)
    st.markdown("- 📆 Sundays and Saturdays show significantly higher renewable share, reflecting lower industrial demand.")

    # Weekend vs Weekday
    weekend = df.groupby('is_weekend')['High_Renewable_Share'].mean().reset_index()
    weekend['Type'] = weekend['is_weekend'].map({False: 'Weekday', True: 'Weekend'})
    fig9_3 = px.bar(weekend, x='Type', y='High_Renewable_Share', width=500,
                   title='🧭 Renewable Dominance: Weekday vs Weekend',
                   labels={'High_Renewable_Share': 'Share > 50%'})
    st.plotly_chart(fig9_3, use_container_width=True)
    st.markdown("""
    📌 **Conclusion:**  
    On weekends, electricity demand is typically lower — especially from industrial and commercial sectors.
    This reduced demand allows available renewable generation (solar, wind, etc.) to supply a larger share of total consumption.  
    → **Renewables exceed 50% more frequently on weekends.**
    """)

    # Monthly pattern
    monthly = df.groupby('Month')['High_Renewable_Share'].mean().reset_index()
    fig9_4 = px.line(monthly.sort_values(by="High_Renewable_Share"),
                    x='Month', y='High_Renewable_Share',
                    title='📈 Renewable Dominance by Month',
                    labels={'High_Renewable_Share': 'Share > 50%'})
    st.plotly_chart(fig9_4, use_container_width=True)
    st.markdown("""
    🔍 **Key Observations:**
    - 🌞 Lowest renewable shares: **August–September** (~22–23%)
    - 🌧️ Steady rise in **October–November**, peaking in **March–May** (~34–35%)

    The trend reveals:
    - Lower dominance in warmer months → **higher demand** (cooling) + **less wind**
    - Higher dominance in colder months → **stronger wind** + **moderate spring solar**
    """)


                            # ---------------------- ML PREDICTIONS ---------------------- #
elif selected == "🤖 ML Predictions":

    import streamlit as st
    import joblib
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    Load_predict = joblib.load("Load Pipeline.h5")
    Model2 = joblib.load("Model2.h5")
    input_columns = joblib.load("input.joblib")

    name_mapping = {
        'Biomass_Generation': 'Biomass Generation',
        'Hydropower_Generation': 'Hydropower Generation',
        'Wind_Offshore_Generation': 'Wind Offshore Generation',
        'Wind_Onshore_Generation': 'Wind Onshore Generation',
        'Solar_PV_Generation': 'Solar PV Generation',
        'Other_Renewable_Generation': 'Other Renewable Generation',
        'Nuclear_Generation': 'Nuclear Generation',
        'Lignite_Generation': 'Lignite Generation',
        'Hard_Coal_Generation': 'Hard Coal Generation',
        'Fossil_Gas_Generation': 'Fossil Gas Generation',
        'Pumped_Storage_Supply': 'Pumped Storage Supply',
        'Other_Conventional_Generation': 'Other Conventional Generation',

        'Temperature_C': 'Temperature (°C)',
        'Dew_Point_C': 'Dew Point (°C)',
        'Relative_Humidity': 'Relative Humidity (%)',
        'Precipitation_mm': 'Precipitation (mm)',
        'Snow_Depth_mm': 'Snow Depth (mm)',
        'Wind_Direction_deg': 'Wind Direction (°)',
        'Wind_Speed_mps': 'Wind Speed (m/s)',
        'Wind_Gust_mps': 'Wind Gust (m/s)',
        'Pressure_hPa': 'Pressure (hPa)',
        'Sunshine_Duration_min': 'Sunshine Duration (min)',

        'Month': 'Month',
        'hour': 'hour',
        'is_weekend': 'is_weekend',
        'is_holiday': 'is_holiday',

        'Total_generation': 'Actually total generation',
        'Total_renewable_generation': 'Total_renewable_generation',
        'month_sin': 'month_sin',
        'month_cos': 'month_cos',
        'hour_sin': 'hour_sin',
        'hour_cos': 'hour_cos',
        'Load': 'Load',
        "Weather_Group": "Weather Group"
    }

    def encode_time_features(month: int, hour: int):
        month_rad = 2 * np.pi * (month - 1) / 12
        hour_rad = 2 * np.pi * hour / 24
        return {
            "month_sin": np.sin(month_rad),
            "month_cos": np.cos(month_rad),
            "hour_sin": np.sin(hour_rad),
            "hour_cos": np.cos(hour_rad),
        }

    def predction(**safe_inputs):
        original_inputs = {name_mapping.get(k, k): v for k, v in safe_inputs.items()}
        month = original_inputs.pop("Month")
        hour = original_inputs.pop("hour")
        original_inputs.update(encode_time_features(month, hour))

        renewable_sources = [
            'Biomass Generation', 'Hydropower Generation', 'Wind Offshore Generation',
            'Wind Onshore Generation', 'Solar PV Generation', 'Other Renewable Generation'
        ]
        original_inputs['Total_renewable_generation'] = sum(original_inputs[i] for i in renewable_sources)

        all_sources = renewable_sources + [
            'Nuclear Generation', 'Lignite Generation', 'Hard Coal Generation',
            'Fossil Gas Generation', 'Pumped Storage Supply', 'Other Conventional Generation'
        ]
        original_inputs['Actually total generation'] = sum(original_inputs[i] for i in all_sources)

        ordered_data = [original_inputs[col] for col in input_columns if col not in ['Month', 'hour']]
        test_df = pd.DataFrame([ordered_data], columns=[col for col in input_columns if col not in ['Month', 'hour']])

        prediction = Load_predict.predict(test_df)[0]
        test_df["Load"] = prediction

        percent = round((original_inputs['Total_renewable_generation'] / prediction) * 100, 2)
        prediction2 = Model2.predict(test_df)[0]
        renewable_status = "HIGH (≥ 50%)" if prediction2 == 1 else "LOW (< 50%)"

        return prediction, percent, renewable_status

    st.title("Energy Load and Renewable Share Forecast")

    with st.expander("Input Panel", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            biomass = st.number_input("Biomass Generation", min_value=0.0)
            hydro = st.number_input("Hydropower Generation", min_value=0.0)
            wind_offshore = st.number_input("Wind Offshore Generation", min_value=0.0)
            wind_onshore = st.number_input("Wind Onshore Generation", min_value=0.0)
        with col2:
            solar_pv = st.number_input("Solar PV Generation", min_value=0.0)
            other_renew = st.number_input("Other Renewable Generation", min_value=0.0)
            nuclear = st.number_input("Nuclear Generation", min_value=0.0)
            lignite = st.number_input("Lignite Generation", min_value=0.0)
        with col3:
            hard_coal = st.number_input("Hard Coal Generation", min_value=0.0)
            fossil_gas = st.number_input("Fossil Gas Generation", min_value=0.0)
            pumped_storage = st.number_input("Pumped Storage Supply", min_value=0.0)
            other_conventional = st.number_input("Other Conventional Generation", min_value=0.0)

        col4, col5, col6 = st.columns(3)
        with col4:
            temperature = st.slider("Temperature (°C)", -30.0, 45.0, 15.0)
            dew_point = st.slider("Dew Point (°C)", -30.0, 30.0, 10.0)
            humidity = st.slider("Relative Humidity (%)", 0, 100, 50)
        with col5:
            precip = st.slider("Precipitation (mm)", 0.0, 100.0, 1.0)
            snow = st.slider("Snow Depth (mm)", 0.0, 500.0, 0.0)
            wind_dir = st.slider("Wind Direction (°)", 0, 360, 180)
        with col6:
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 40.0, 5.0)
            wind_gust = st.slider("Wind Gust (m/s)", 0.0, 60.0, 10.0)
            pressure = st.slider("Pressure (hPa)", 950.0, 1050.0, 1013.0)
            sunshine = st.slider("Sunshine Duration (min)", 0, 720, 300)

        col7, col8 = st.columns(2)
        with col7:
            month = st.selectbox("Month", list(range(1, 13)))
            hour = st.selectbox("Hour", list(range(0, 24)))
        with col8:
            is_weekend = st.checkbox("Weekend?")
            is_holiday = st.checkbox("Holiday?")
            weather_group = st.selectbox("Weather Group", ['Cloudy', 'Rain', 'Snow', 'Clear', 'Storm', 'Extreme'])

    if st.button("Predict"):
        with st.spinner("Calculating predictions..."):
            prediction, percent, renewable_status = predction(
                Biomass_Generation=biomass,
                Hydropower_Generation=hydro,
                Wind_Offshore_Generation=wind_offshore,
                Wind_Onshore_Generation=wind_onshore,
                Solar_PV_Generation=solar_pv,
                Other_Renewable_Generation=other_renew,
                Nuclear_Generation=nuclear,
                Lignite_Generation=lignite,
                Hard_Coal_Generation=hard_coal,
                Fossil_Gas_Generation=fossil_gas,
                Pumped_Storage_Supply=pumped_storage,
                Other_Conventional_Generation=other_conventional,
                Temperature_C=temperature,
                Dew_Point_C=dew_point,
                Relative_Humidity=humidity,
                Precipitation_mm=precip,
                Snow_Depth_mm=snow,
                Wind_Direction_deg=wind_dir,
                Wind_Speed_mps=wind_speed,
                Wind_Gust_mps=wind_gust,
                Pressure_hPa=pressure,
                Sunshine_Duration_min=sunshine,
                Month=month,
                hour=hour,
                is_weekend=int(is_weekend),
                is_holiday=int(is_holiday),
                Weather_Group=weather_group,
            )

            hour_str = f"{hour:02d}:00"
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Load Estimation (MW)")
                fig_Load = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    title={'text': hour_str},
                    gauge={
                        'axis': {'range': [0, max(250, prediction * 1.3)]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, prediction * 0.6], 'color': "lightgreen"},
                            {'range': [prediction * 0.6, prediction * 0.9], 'color': "gold"},
                            {'range': [prediction * 0.9, max(250, prediction * 1.3)], 'color': "red"},
                        ],
                    }
                ))
                st.plotly_chart(fig_Load, use_container_width=True)

            with col2:
                st.subheader("Renewable Share")
                fig_Renewable = px.pie(
                    values=[percent, 100 - percent],
                    names=["Renewables", "Others"],
                    hole=0.6,
                    color_discrete_sequence=["green", "gray"]
                )
                fig_Renewable.update_traces(textinfo="percent+label")
                st.plotly_chart(fig_Renewable, use_container_width=True)

            st.markdown("### Summary")
            st.info(f"At **{hour_str}**, the predicted load is **{round(prediction, 2)} MW**, "
                    f"with a renewable contribution of **{percent}%** → **{renewable_status}**.")



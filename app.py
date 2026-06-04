import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# LOAD FILES
# -----------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

df = pd.read_csv("HR-Employee-Attrition.csv")

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Prediction"]
)

# ===================================
# DASHBOARD PAGE
# ===================================
if page == "Dashboard":

    st.title("📊 HR Analytics Dashboard")

    total_employees = len(df)
    attrition_count = len(df[df["Attrition"] == "Yes"])

    attrition_rate = round(
        (attrition_count / total_employees) * 100,
        2
    )

    avg_income = round(df["MonthlyIncome"].mean(), 0)
    avg_age = round(df["Age"].mean(), 1)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Employees", total_employees)
    col2.metric("Attrition Rate", f"{attrition_rate}%")
    col3.metric("Avg Income", f"${avg_income:,.0f}")
    col4.metric("Avg Age", avg_age)

    st.divider()

    # Attrition Pie Chart
    col1, col2 = st.columns(2)

    with col1:

        fig1 = px.pie(
            df,
            names="Attrition",
            title="Employee Attrition",
            color_discrete_sequence=["#00B4D8", "#FF6B6B"]
        )

        fig1.update_layout(
            paper_bgcolor="#111827",
            font_color="white"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        dept = (
            df.groupby("Department")
            .size()
            .reset_index(name="Employees")
        )

        fig2 = px.bar(
            dept,
            x="Department",
            y="Employees",
            title="Employees by Department",
            color_discrete_sequence=["#4ECDC4"]
        )

        fig2.update_layout(
            plot_bgcolor="#111827",
            paper_bgcolor="#111827",
            font_color="white"
        )

        fig2.update_xaxes(
            title_font=dict(color="yellow"),
            tickfont=dict(color="white")
        )

        fig2.update_yaxes(
            title_font=dict(color="yellow"),
            tickfont=dict(color="white")
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Income Distribution
    st.subheader("Income Distribution")

    fig = px.histogram(
        df,
        x="MonthlyIncome",
        nbins=30,
        title="Monthly Income Distribution",
        color_discrete_sequence=["#FFA502"]
    )

    fig.update_layout(
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Age Distribution
    st.subheader("Age Distribution")

    fig = px.histogram(
        df,
        x="Age",
        nbins=20,
        title="Age Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Gender Analysis
    st.subheader("Gender Analysis")

    fig = px.histogram(
        df,
        x="Gender",
        color="Attrition",
        barmode="group",
        title="Attrition by Gender",
        color_discrete_sequence=["#00B4D8", "#FF6B6B"]
        )

    fig.update_layout(
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Attrition by Department

    dept_attr = (
    df.groupby(["Department", "Attrition"])
    .size()
    .reset_index(name="Count")
    )

    fig = px.bar(
        dept_attr,
        x="Department",
        y="Count",
        color="Attrition",
        barmode="group",
        title="Attrition by Department"
)

    st.plotly_chart(fig, use_container_width=True)

    # Top 5 Job Roles Chart

    role_attr = (
    df[df["Attrition"]=="Yes"]
    .groupby("JobRole")
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
    .head(5)
    )

    fig = px.bar(
    role_attr,
    x="JobRole",
    y="Count",
    title="Top 5 Job Roles with Highest Attrition",
    color_discrete_sequence=["#FF4757"]
    )

    fig.update_layout(
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    
    

# ===================================
# PREDICTION PAGE
# ===================================
elif page == "Prediction":

    st.title("🤖 Employee Attrition Prediction")

    age = st.number_input("Age", 18, 60, 30)

    daily_rate = st.number_input(
        "Daily Rate",
        100,
        2000,
        800
    )

    distance = st.number_input(
        "Distance From Home",
        1,
        30,
        5
    )

    monthly_income = st.number_input(
        "Monthly Income",
        1000,
        50000,
        5000
    )

    years_company = st.number_input(
        "Years At Company",
        0,
        40,
        5
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

    department = st.selectbox(
        "Department",
        [
            "Research & Development",
            "Sales",
            "Human Resources"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    overtime = st.selectbox(
        "OverTime",
        [
            "Yes",
            "No"
        ]
    )

    if st.button("Predict Attrition"):

        sample = {
            "Age": age,
            "DailyRate": daily_rate,
            "DistanceFromHome": distance,
            "MonthlyIncome": monthly_income,
            "YearsAtCompany": years_company,
            "BusinessTravel": business_travel,
            "Department": department,
            "Gender": gender,
            "MaritalStatus": marital_status,
            "OverTime": overtime
        }

        input_df = pd.DataFrame([sample])

        input_df = pd.get_dummies(input_df)

        final_df = pd.DataFrame(
            columns=columns
        )

        for col in input_df.columns:
            if col in final_df.columns:
                final_df[col] = input_df[col]

        final_df = final_df.fillna(0)

        scaled_data = scaler.transform(final_df)

        prediction = model.predict(
            scaled_data
        )[0]

        probability = model.predict_proba(
            scaled_data
        )[0]

        if prediction == 1:
            st.error(
                "⚠️ Employee Likely To Leave"
            )
        else:
            st.success(
                "✅ Employee Likely To Stay"
            )

        st.write(
            f"Attrition Probability: {probability[1]*100:.2f}%"
        )

        st.progress(
            float(probability[1])
        )
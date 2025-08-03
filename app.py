import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Tiêu đề
st.title("💎 Diamond Sales Analysis App")
st.markdown("A simple data science solution using Streamlit for data visualization and price prediction.")

# 2. Đọc dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
    return df

df = load_data()

# 3. Hiển thị dữ liệu gốc
if st.checkbox("Show raw data"):
    st.subheader("Raw Dataset")
    st.write(df.head())

# 4. Bộ lọc dữ liệu
st.sidebar.header("Filter Options")
cut_filter = st.sidebar.multiselect("Select Cut", df['cut'].unique(), default=df['cut'].unique())
color_filter = st.sidebar.multiselect("Select Color", df['color'].unique(), default=df['color'].unique())

filtered_df = df[(df['cut'].isin(cut_filter)) & (df['color'].isin(color_filter))]

st.subheader("Filtered Dataset")
st.write(filtered_df.head())

# 5. Biểu đồ trực quan
st.subheader("📊 Visualization")

chart = st.selectbox("Choose a chart to display", ["Price by Cut", "Price vs Carat", "Boxplot of Price by Color"])

if chart == "Price by Cut":
    fig = plt.figure(figsize=(8,4))
    sns.barplot(data=filtered_df, x='cut', y='price')
    st.pyplot(fig)

elif chart == "Price vs Carat":
    fig = plt.figure(figsize=(8,4))
    sns.scatterplot(data=filtered_df, x='carat', y='price', hue='cut')
    st.pyplot(fig)

elif chart == "Boxplot of Price by Color":
    fig = plt.figure(figsize=(8,4))
    sns.boxplot(data=filtered_df, x='color', y='price')
    st.pyplot(fig)

# 6. Dự đoán giá bằng Linear Regression
st.subheader("📈 Predict Price using Linear Regression")

# Encode categorical variables
df_encoded = pd.get_dummies(df[['carat', 'cut', 'color', 'clarity']])
df_encoded['price'] = df['price']

X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred, squared=True) ** 0.5:.2f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

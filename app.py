import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cấu hình giao diện
st.set_page_config(page_title="Diamond Sales Analysis", layout="wide")

# Tải dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
    return df

df = load_data()

# Hiển thị dữ liệu
st.title("💎 Diamond Sales Data Analysis")
st.write("### Preview of the Dataset")
st.dataframe(df.head())

# Biểu đồ 1: Tổng doanh số theo tháng
st.write("### 📊 Total Sales by Month")
monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
fig1 = plt.figure(figsize=(10,5))
sns.lineplot(data=monthly_sales, x='Month', y='Sales', marker='o')
plt.xticks(rotation=45)
plt.title("Monthly Sales Trend")
st.pyplot(fig1)

# Biểu đồ 2: Doanh số theo sản phẩm
st.write("### 📈 Sales by Product")
product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
fig2 = plt.figure(figsize=(10,5))
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.xticks(rotation=45)
plt.title("Sales by Product")
st.pyplot(fig2)

# Biểu đồ 3: Số lượng bán theo sản phẩm
st.write("### 📦 Quantity Sold by Product")
fig3 = plt.figure(figsize=(10,5))
sns.barplot(data=df, x='Product', y='Quantity')
plt.xticks(rotation=45)
plt.title("Quantity Sold per Product")
st.pyplot(fig3)

# Biểu đồ 4: Biểu đồ phân phối giá bán
st.write("### 💰 Price Distribution")
fig4 = plt.figure(figsize=(8,5))
sns.histplot(df['Price'], bins=20, kde=True)
plt.title("Price Distribution")
st.pyplot(fig4)

# Biểu đồ 5: Tương quan giá và số lượng
st.write("### 📉 Correlation between Price and Quantity")

# Xử lý dữ liệu: loại bỏ các dòng có NaN trong Price hoặc Quantity
df_corr = df[['Price', 'Quantity']].dropna()

# Kiểm tra nếu dữ liệu hợp lệ
if df_corr.empty:
    st.warning("Không có dữ liệu hợp lệ để hiển thị biểu đồ tương quan.")
else:
    fig5 = plt.figure(figsize=(8,5))
    sns.scatterplot(data=df_corr, x='Price', y='Quantity')
    plt.title("Price vs Quantity Sold")
    st.pyplot(fig5)

# Mô hình dự báo doanh số bằng Linear Regression
st.write("## 🤖 Sales Prediction Model")

# Lựa chọn biến đầu vào
features = ['Price', 'Quantity']
target = 'Sales'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

st.write(f"**Mean Squared Error (MSE)**: {mse:.2f}")

# Dự đoán thử từ input người dùng
st.write("### 🧪 Try Sales Prediction")

price_input = st.number_input("Enter Price", value=100.0)
quantity_input = st.number_input("Enter Quantity", value=1)

predicted_sales = model.predict([[price_input, quantity_input]])
st.success(f"📊 Predicted Sales: {predicted_sales[0]:.2f}")

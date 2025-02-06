# ตัวอย่างที่ 1 Pie Sales Example 
# หน้าที่ 5 Unit 7: Multiple Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
from sklearn.metrics import r2_score

# สร้าง DataFrame จากข้อมูลในรูป
# การสร้างค่าลำดับสำหรับคอลัมน์ Week ที่เริ่มต้นจาก 1 และสิ้นสุดที่ 15 (แต่ไม่รวม 16) โดยใช้ฟังก์ชัน range() ใน Python
data = {
    "Week": range(1, 16),
    "Pie Sales": [350, 460, 350, 430, 350, 380, 430, 470, 450, 490, 340, 300, 440, 450, 300],
    "Price ($)": [5.50, 7.50, 8.00, 8.00, 6.80, 7.50, 4.50, 6.40, 7.00, 5.00, 7.20, 7.90, 5.90, 5.00, 7.00],
    "Advertising ($100s)": [3.3, 3.3, 3.0, 4.5, 3.0, 4.0, 3.0, 3.7, 3.5, 4.0, 3.5, 3.2, 4.0, 3.5, 2.7],
}

df = pd.DataFrame(data)

# ตัวแปรอิสระ (Price, Advertising)
X = df[["Price ($)", "Advertising ($100s)"]]

# ตัวแปรตาม (Pie Sales)
y = df["Pie Sales"]

# เพิ่มคอลัมน์คงที่สำหรับโมเดล
X_const = sm.add_constant(X)

# สร้างโมเดลด้วย Statsmodels
model = sm.OLS(y, X_const).fit()

# ทำนายค่าจากโมเดล
y_pred = model.predict(X_const)

# คำนวณ R-squared
r2 = r2_score(y, y_pred)

# แสดง Summary ของโมเดล
print(model.summary())

# ดึงค่าคงที่และสัมประสิทธิ์ (Coefficients) ของโมเดล
intercept = model.params['const']
price_coef = model.params['Price ($)']
advertising_coef = model.params['Advertising ($100s)']

# สร้างกราฟแสดงผล
plt.figure(figsize=(12, 6))
plt.scatter(y, y_pred, color="blue", label="Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2, label="Ideal Fit")
plt.title("Actual vs Predicted Pie Sales")
plt.xlabel("Actual Pie Sales")
plt.ylabel("Predicted Pie Sales")
plt.legend()
plt.grid()
plt.show()

# พิมพ์สมการของโมเดล
print("\nสมการของโมเดล:")
print(f"Sales = {intercept:.2f} + ({price_coef:.2f} * Price) + ({advertising_coef:.2f} * Advertising)")
# แสดงค่า R-squared
print(f"R-squared: {r2}")

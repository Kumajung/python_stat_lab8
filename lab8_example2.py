# ตัวอย่างที่ 2 "การวิเคราะห์ปัจจัยที่ส่งผลต่อยอดขายพาย โดยพิจารณาราคา การโฆษณา ส่วนลด และความพึงพอใจของลูกค้า"
# รายละเอียดโจทย์:
# วิเคราะห์ปัจจัยทั้ง 4 ได้แก่:

# Price: ราคาของพายในแต่ละสัปดาห์
# Advertising: การลงทุนในการโฆษณา (ในหน่วย $100s)
# Discount: ส่วนลดที่ให้ลูกค้า (ในหน่วย $)
# Customer Satisfaction: ระดับความพึงพอใจของลูกค้า (คะแนน 1-5)
# โดยมีเป้าหมายเพื่อ:

# ศึกษาความสัมพันธ์ของปัจจัยทั้ง 4 ต่อยอดขายพายในแต่ละสัปดาห์ (Pie Sales)
# สร้างสมการการพยากรณ์ยอดขายจากตัวแปรทั้ง 4 เพื่อช่วยสนับสนุนการตัดสินใจทางธุรกิจ
# สมการที่ได้จะช่วยให้ผู้จัดการสามารถกำหนดราคาพาย ปรับการโฆษณา และส่วนลด รวมถึงประเมินความพึงพอใจของลูกค้า 
# เพื่อเพิ่มยอดขายในอนาคตอย่างมีประสิทธิภาพ

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
from sklearn.metrics import r2_score

data = {
    "Week": range(1, 16),
    "Pie Sales": [350, 460, 350, 430, 350, 380, 430, 470, 450, 490, 340, 300, 440, 450, 300],
    "Price ($)": [5.50, 7.50, 8.00, 8.00, 6.80, 7.50, 4.50, 6.40, 7.00, 5.00, 7.20, 7.90, 5.90, 5.00, 7.00],
    "Advertising ($100s)": [3.3, 3.3, 3.0, 4.5, 3.0, 4.0, 3.0, 3.7, 3.5, 4.0, 3.5, 3.2, 4.0, 3.5, 2.7],
    "Discount ($)": [0.50, 1.00, 0.30, 0.25, 0.40, 0.60, 0.70, 0.35, 0.45, 0.50, 0.60, 0.40, 0.50, 0.30, 0.25],
    "Customer Satisfaction": [4.5, 4.7, 4.2, 4.3, 4.1, 4.6, 4.4, 4.8, 4.5, 4.9, 4.0, 3.8, 4.5, 4.6, 3.9]
}

df = pd.DataFrame(data)
# ตัวแปรอิสระ (Price, Advertising, Discount, Customer Satisfaction)
X = df[["Price ($)", "Advertising ($100s)", "Discount ($)", "Customer Satisfaction"]]

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
discount_coef = model.params['Discount ($)']
satisfaction_coef = model.params['Customer Satisfaction']

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
print(f"Sales = {intercept:.2f} + ({price_coef:.2f} * Price) + ({advertising_coef:.2f} * Advertising) + "
      f"({discount_coef:.2f} * Discount) + ({satisfaction_coef:.2f} * Customer Satisfaction)")

# แสดงค่า R-squared
print(f"R-squared: {r2}")


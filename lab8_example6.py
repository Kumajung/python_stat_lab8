# ตัวอย่างที่ 6 การพยากรณ์ราคาบ้าน
# บริษัทอสังหาริมทรัพย์ต้องการพยากรณ์ราคาบ้านในเขตเมือง โดยใช้ตัวแปรดังนี้

# X1 = พื้นที่ใช้สอยของบ้าน (ตารางเมตร)
# X2 = จำนวนห้องนอน
# X3 = ระยะทางจากตัวเมือง (กิโลเมตร)
# Y = ราคาบ้าน (บาท)
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# อ่านข้อมูลจากไฟล์ .csv
file_path = "/kaggle/input/dataset001/house_prices.csv"  # เปลี่ยนชื่อไฟล์เป็นชื่อไฟล์ของคุณ
df = pd.read_csv(file_path)

# ตรวจสอบข้อมูล
print(df.head())

# ตัวแปรอิสระ (Living Area, Bedrooms, Distance to City Center)
X = df[["Living Area", "Bedrooms", "Distance to City Center"]]

# ตัวแปรตาม (Price)
y = df["Price"]

# เพิ่มคอลัมน์คงที่
X_const = sm.add_constant(X)

# สร้างโมเดลด้วย Statsmodels
model = sm.OLS(y, X_const).fit()

# ทำนายค่า
y_pred = model.predict(X_const)

# คำนวณ R-squared
r2 = r2_score(y, y_pred)

# แสดง Summary ของโมเดล
print(model.summary())

# ดึงค่าคงที่และสัมประสิทธิ์
intercept = model.params['const']
living_area_coef = model.params['Living Area']
bedrooms_coef = model.params['Bedrooms']
distance_coef = model.params['Distance to City Center']

# สร้างกราฟเปรียบเทียบ
plt.figure(figsize=(12, 6))
plt.scatter(y, y_pred, color="blue", label="Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2, label="Ideal Fit")
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual House Prices (Baht)")
plt.ylabel("Predicted House Prices (Baht)")
plt.legend()
plt.grid()
plt.show()

# พิมพ์สมการของโมเดล
print("\nสมการของโมเดล:")
print(f"Price = {intercept:.2f} + ({living_area_coef:.2f} * Living Area) + ({bedrooms_coef:.2f} * Bedrooms) + "
      f"({distance_coef:.2f} * Distance to City Center)")

# แสดงค่า R-squared
print(f"\nR-squared: {r2:.2f}")

# ตัวอย่างที่ 8 บริษัทผลิตภัณฑ์อุปโภคบริโภคต้องการคาดการณ์ยอดขาย (Y) ของสินค้าหนึ่งรายการ โดยใช้ตัวแปรดังนี้
# X1 = งบโฆษณา (บาท)
# X2 = จำนวนร้านค้าที่วางจำหน่าย
# X3 = ราคาขายต่อหน่วย (บาท)
# Y = ยอดขายรายเดือน (หน่วย)
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# อ่านข้อมูลจากไฟล์ .csv
file_path = "/kaggle/input/dataset001/product_sales.csv"  # เปลี่ยนเป็น path ที่เหมาะสม
df = pd.read_csv(file_path)

# ตรวจสอบข้อมูล
print(df.head())

# ตัวแปรอิสระ (Advertising Budget, Number of Stores, Price per Unit)
X = df[["Advertising Budget (Baht)", "Number of Stores", "Price per Unit (Baht)"]]

# ตัวแปรตาม (Monthly Sales)
y = df["Monthly Sales (Units)"]

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
advertising_coef = model.params['Advertising Budget (Baht)']
stores_coef = model.params['Number of Stores']
price_coef = model.params['Price per Unit (Baht)']

# สร้างกราฟเปรียบเทียบ
plt.figure(figsize=(12, 6))
plt.scatter(y, y_pred, color="blue", label="Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2, label="Ideal Fit")
plt.title("Actual vs Predicted Monthly Sales")
plt.xlabel("Actual Monthly Sales (Units)")
plt.ylabel("Predicted Monthly Sales (Units)")
plt.legend()
plt.grid()
plt.show()

# พิมพ์สมการของโมเดล
print("\nสมการของโมเดล:")
print(f"Monthly Sales = {intercept:.2f} + ({advertising_coef:.2f} * Advertising Budget) + "
      f"({stores_coef:.2f} * Number of Stores) + ({price_coef:.2f} * Price per Unit)")

# แสดงค่า R-squared
print(f"\nR-squared: {r2:.2f}")

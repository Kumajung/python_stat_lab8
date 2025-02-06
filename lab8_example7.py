# ตัวอย่างที่ 7 การวิเคราะห์ปัจจัยที่มีผลต่อผลการเรียนของนักเรียน
# นักวิจัยต้องการวิเคราะห์ว่าปัจจัยใดมีผลกระทบต่อผลการเรียนของนักเรียน (GPA) โดยใช้ตัวแปรดังนี้

# X1 = จำนวนชั่วโมงที่นักเรียนใช้ในการเรียนรู้ต่อสัปดาห์
# X2 = ระดับรายได้ของครอบครัว (บาทต่อเดือน)
# X3 = จำนวนชั่วโมงที่ใช้เล่นเกมต่อสัปดาห์
# Y = ค่าเฉลี่ยผลการเรียน (GPA)
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# อ่านข้อมูลจากไฟล์ .csv
file_path = "/kaggle/input/dataset001/student_performance.csv"  # เปลี่ยนเป็น path ของคุณ
df = pd.read_csv(file_path)

# ตรวจสอบข้อมูล
print(df.head())

# ตัวแปรอิสระ (Study Hours, Family Income, Gaming Hours)
X = df[["Study Hours Per Week", "Family Income (Baht/Month)", "Gaming Hours Per Week"]]

# ตัวแปรตาม (GPA)
y = df["GPA"]

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
study_hours_coef = model.params['Study Hours Per Week']
family_income_coef = model.params['Family Income (Baht/Month)']
gaming_hours_coef = model.params['Gaming Hours Per Week']

# สร้างกราฟเปรียบเทียบ
plt.figure(figsize=(12, 6))
plt.scatter(y, y_pred, color="blue", label="Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2, label="Ideal Fit")
plt.title("Actual vs Predicted GPA")
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.legend()
plt.grid()
plt.show()

# พิมพ์สมการของโมเดล
print("\nสมการของโมเดล:")
print(f"GPA = {intercept:.2f} + ({study_hours_coef:.2f} * Study Hours Per Week) + "
      f"({family_income_coef:.2f} * Family Income) + ({gaming_hours_coef:.2f} * Gaming Hours Per Week)")

# แสดงค่า R-squared
print(f"\nR-squared: {r2:.2f}")

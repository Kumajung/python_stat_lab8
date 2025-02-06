# โจทย์ที่ 1: การคาดการณ์อัตราการลาออกของพนักงาน (Employee Turnover Prediction)
# บริษัทเทคโนโลยีต้องการวิเคราะห์ปัจจัยที่ส่งผลต่อการลาออกของพนักงาน โดยใช้ตัวแปรดังนี้

# X1 = ระยะเวลาการทำงานในบริษัท (ปี)
# X2 = ระดับความพึงพอใจในงาน (คะแนนจาก 1-10)
# X3 = ความถี่ในการทำงานล่วงเวลา (ชั่วโมงต่อสัปดาห์)
# Y = ความน่าจะเป็นที่พนักงานจะลาออก (เป็นเปอร์เซ็นต์)

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# กำหนดข้อมูล
data = {
    "Years at Company": [2, 3, 1, 5, 10, 7, 4, 8, 6, 3,
                         2, 4, 9, 7, 5, 6, 3, 8, 2, 4,
                         5, 6, 7, 9, 2, 1, 3, 8, 6, 4,
                         7, 10, 9, 5, 4, 3, 6, 8, 2, 7,
                         9, 5, 6, 7, 4, 3, 2, 10, 8, 9],
    "Job Satisfaction (1-10)": [7, 8, 5, 9, 10, 6, 7, 8, 5, 7,
                                6, 8, 9, 7, 8, 6, 5, 7, 6, 8,
                                9, 8, 7, 6, 5, 4, 8, 9, 6, 7,
                                5, 10, 9, 8, 7, 6, 5, 9, 8, 7,
                                6, 9, 7, 6, 8, 9, 7, 8, 10, 6],
    "Overtime Hours per Week": [5, 10, 3, 15, 20, 12, 7, 18, 10, 8,
                                 6, 12, 14, 9, 11, 13, 7, 16, 5, 12,
                                 15, 11, 13, 17, 4, 3, 10, 14, 12, 9,
                                 13, 18, 16, 12, 11, 8, 15, 17, 6, 14,
                                 16, 13, 12, 11, 8, 6, 5, 20, 14, 16],
    "Turnover Probability (%)": [15, 25, 10, 35, 50, 40, 20, 45, 30, 25,
                                  15, 35, 50, 30, 40, 38, 25, 45, 12, 35,
                                  40, 38, 45, 50, 10, 8, 28, 48, 33, 25,
                                  40, 50, 47, 39, 35, 28, 42, 49, 20, 41,
                                  48, 43, 35, 38, 27, 20, 18, 50, 46, 48]
}

# สร้าง DataFrame
df = pd.DataFrame(data)

# ตัวแปรอิสระ (Years at Company, Job Satisfaction, Overtime Hours per Week)
X = df[["Years at Company", "Job Satisfaction (1-10)", "Overtime Hours per Week"]]

# ตัวแปรตาม (Turnover Probability)
y = df["Turnover Probability (%)"]

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
years_coef = model.params['Years at Company']
satisfaction_coef = model.params['Job Satisfaction (1-10)']
overtime_coef = model.params['Overtime Hours per Week']

# สร้างกราฟเปรียบเทียบ
plt.figure(figsize=(12, 6))
plt.scatter(y, y_pred, color="blue", label="Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2, label="Ideal Fit")
plt.title("Actual vs Predicted Turnover Probability")
plt.xlabel("Actual Turnover Probability (%)")
plt.ylabel("Predicted Turnover Probability (%)")
plt.legend()
plt.grid()
plt.show()

# พิมพ์สมการของโมเดล
print("\nสมการของโมเดล:")
print(f"Turnover Probability (%) = {intercept:.2f} + ({years_coef:.2f} * Years at Company) + "
      f"({satisfaction_coef:.2f} * Job Satisfaction) + ({overtime_coef:.2f} * Overtime Hours per Week)")

# แสดงค่า R-squared
print(f"\nR-squared: {r2:.2f}")
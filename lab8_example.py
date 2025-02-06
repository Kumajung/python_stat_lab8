# ตัวอย่างที่ 5 
# "การวิเคราะห์ปัจจัยที่ส่งผลต่อจำนวนชั่วโมงการดู Netflix ต่อวัน โดยพิจารณาปัจจัยอิสระ ได้แก่ อายุ ชั่วโมงการนอนหลับ ระดับความเครียด และอิทธิพลจากสื่อ"

# รายละเอียดโจทย์:
# ตัวแปรตาม (Dependent Variable):

# Hours Watching Netflix: จำนวนชั่วโมงการดู Netflix ต่อวัน
# ตัวแปรอิสระ (Independent Variables):

# Age: อายุของผู้ดู Netflix (ปี)
# Sleep Hours: จำนวนชั่วโมงการนอนหลับต่อวัน
# Stress Level (1-10): ระดับความเครียด (1 = ต่ำสุด, 10 = สูงสุด)
# Media Influence (1-10): อิทธิพลจากสื่อ (1 = น้อยที่สุด, 10 = มากที่สุด)

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# กำหนดข้อมูล DataFrame (n=35)
data = {
    "Age": [18, 20, 25, 22, 30, 19, 28, 24, 21, 26,
            29, 22, 23, 27, 19, 20, 24, 22, 25, 21,
            18, 30, 28, 26, 29, 23, 22, 19, 24, 20,
            21, 22, 23, 24, 25],
    "Sleep Hours": [7.5, 6.0, 5.5, 8.0, 6.5, 7.0, 5.0, 7.8, 6.2, 6.0,
                    5.8, 7.4, 6.3, 6.9, 7.1, 6.5, 7.3, 6.0, 5.7, 7.0,
                    6.8, 5.9, 5.2, 6.7, 5.4, 6.1, 6.3, 7.2, 5.6, 6.4,
                    7.0, 6.2, 5.8, 6.5, 7.0],
    "Stress Level": [5, 7, 6, 8, 5, 4, 7, 6, 8, 5,
                     6, 7, 6, 5, 4, 7, 6, 5, 8, 7,
                     6, 4, 6, 5, 7, 6, 5, 8, 7, 5,
                     6, 6, 7, 6, 5],
    "Media Influence": [7, 8, 9, 6, 7, 8, 9, 6, 7, 8,
                        9, 6, 7, 8, 9, 6, 7, 8, 9, 6,
                        7, 8, 9, 6, 7, 8, 9, 6, 7, 8,
                        9, 6, 7, 8, 9],
    "Hours Watching Netflix": [2.5, 3.0, 4.0, 2.8, 3.5, 2.0, 4.2, 3.7, 3.8, 2.9,
                                4.5, 3.6, 4.1, 3.3, 2.4, 3.0, 3.9, 3.5, 4.0, 3.2,
                                3.1, 4.2, 3.8, 4.1, 3.0, 3.4, 3.6, 3.9, 4.0, 3.8,
                                3.7, 3.2, 4.0, 3.6, 3.5]
}

df = pd.DataFrame(data)

# ตัวแปรอิสระ (Age, Sleep Hours, Stress Level, Media Influence)
X = df[["Age", "Sleep Hours", "Stress Level", "Media Influence"]]

# ตัวแปรตาม (Hours Watching Netflix)
y = df["Hours Watching Netflix"]

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
age_coef = model.params['Age']
sleep_coef = model.params['Sleep Hours']
stress_coef = model.params['Stress Level']
media_influence_coef = model.params['Media Influence']

# สร้างกราฟเปรียบเทียบ
plt.figure(figsize=(12, 6))
plt.scatter(y, y_pred, color="blue", label="Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2, label="Ideal Fit")
plt.title("Actual vs Predicted Netflix Watching Hours")
plt.xlabel("Actual Watching Hours")
plt.ylabel("Predicted Watching Hours")
plt.legend()
plt.grid()
plt.show()

# พิมพ์สมการของโมเดล
print("\nสมการของโมเดล:")
print(f"Hours Watching Netflix = {intercept:.2f} + ({age_coef:.2f} * Age) + ({sleep_coef:.2f} * Sleep Hours) + "
      f"({stress_coef:.2f} * Stress Level) + ({media_influence_coef:.2f} * Media Influence)")

# แสดงค่า R-squared
print(f"\nR-squared: {r2:.2f}")

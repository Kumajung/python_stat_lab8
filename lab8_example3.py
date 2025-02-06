# ตัวอย่างที่ 3 
# "การวิเคราะห์ปัจจัยที่ส่งผลต่อจำนวนชั่วโมงการใช้ TikTok ต่อวัน 
# โดยพิจารณาปัจจัยอิสระ 4 ตัว ได้แก่ อายุ รายได้ ระดับการศึกษาของผู้ใช้ และอิทธิพลจากเพื่อน"

# รายละเอียดโจทย์:
# ตัวแปรตาม (Dependent Variable):

# Hours per Day: จำนวนชั่วโมงที่ใช้ TikTok ต่อวัน
# ตัวแปรอิสระ (Independent Variables):

# Age: อายุของผู้ใช้ (ปี)
# Income ($1000s): รายได้ของผู้ใช้ (หน่วยเป็นพันดอลลาร์)
# Education Level: ระดับการศึกษา (1 = ประถมศึกษา, 2 = มัธยมศึกษา, 3 = ปริญญาตรี, 4 = สูงกว่าปริญญาตรี)
# Peer Influence (1-10): คะแนนอิทธิพลจากเพื่อนที่ชักชวนให้ใช้ TikTok (1 = น้อยที่สุด, 10 = มากที่สุด)

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# กำหนดข้อมูล DataFrame
data = {
    "Age": [18, 25, 30, 22, 45, 19, 34, 27, 40, 20,
            23, 31, 29, 37, 24, 21, 26, 33, 38, 28,
            32, 36, 44, 39, 41, 35, 42, 43, 46, 47,
            17, 16, 15, 22, 19, 18, 30, 24, 31, 25,
            23, 21, 20, 22, 26, 34, 28, 27, 29, 33],
    "Income ($1000s)": [3.5, 4.0, 5.5, 3.8, 7.0, 2.5, 6.2, 4.3, 8.5, 3.0,
                        3.9, 5.0, 4.8, 6.5, 4.1, 3.2, 4.4, 6.0, 7.2, 4.7,
                        5.8, 6.3, 7.5, 6.8, 7.3, 5.9, 7.8, 7.9, 8.0, 8.2,
                        2.0, 1.5, 1.2, 3.6, 2.8, 3.1, 5.2, 4.2, 5.1, 3.3,
                        4.0, 3.3, 3.7, 3.4, 4.5, 6.1, 4.6, 4.9, 5.3, 6.4],
    "Education Level": [3, 4, 4, 3, 2, 2, 4, 3, 1, 3,
                        3, 4, 4, 1, 2, 3, 3, 4, 2, 4,
                        4, 1, 1, 2, 1, 4, 2, 1, 1, 1,
                        3, 2, 2, 3, 2, 3, 4, 2, 4, 3,
                        3, 3, 3, 3, 4, 2, 4, 3, 4, 4],
    "Peer Influence": [6, 8, 7, 5, 3, 9, 4, 7, 2, 10,
                       6, 5, 8, 4, 7, 6, 7, 5, 3, 8,
                       7, 4, 3, 3, 4, 5, 3, 3, 2, 2,
                       9, 10, 8, 7, 9, 8, 5, 7, 5, 6,
                       5, 6, 6, 7, 6, 5, 6, 8, 7, 7],
    "Hours per Day": [2.5, 3.2, 2.0, 2.8, 1.5, 4.0, 2.2, 3.0, 1.0, 4.5,
                      3.1, 2.4, 3.5, 2.0, 3.3, 2.6, 3.0, 2.3, 1.8, 3.7,
                      3.2, 2.1, 1.7, 1.9, 1.6, 2.9, 1.5, 1.4, 1.2, 1.3,
                      4.2, 4.5, 4.8, 3.6, 4.1, 4.0, 2.5, 3.4, 2.6, 4.4,
                      3.1, 2.7, 3.2, 2.8, 3.3, 2.4, 3.6, 3.5, 3.2, 2.8]
}

df = pd.DataFrame(data)

# ตัวแปรอิสระ (Age, Income, Education Level, Peer Influence)
X = df[["Age", "Income ($1000s)", "Education Level", "Peer Influence"]]

# ตัวแปรตาม (Hours per Day)
y = df["Hours per Day"]

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
income_coef = model.params['Income ($1000s)']
education_coef = model.params['Education Level']
peer_influence_coef = model.params['Peer Influence']

# สร้างกราฟเปรียบเทียบ
plt.figure(figsize=(12, 6))
plt.scatter(y, y_pred, color="blue", label="Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2, label="Ideal Fit")
plt.title("Actual vs Predicted TikTok Usage (Hours per Day)")
plt.xlabel("Actual TikTok Usage (Hours)")
plt.ylabel("Predicted TikTok Usage (Hours)")
plt.legend()
plt.grid()
plt.show()

# พิมพ์สมการของโมเดล
print("\nสมการของโมเดล:")
print(f"Hours per Day = {intercept:.2f} + ({age_coef:.2f} * Age) + ({income_coef:.2f} * Income) + "
      f"({education_coef:.2f} * Education Level) + ({peer_influence_coef:.2f} * Peer Influence)")

# แสดงค่า R-squared
print(f"\nR-squared: {r2:.2f}")

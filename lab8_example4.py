# ตัวอย่างที่ 4 
# การวิเคราะห์ปัจจัยที่ส่งผลต่อจำนวนชั่วโมงการเล่นเกม ROV ต่อวัน โดยพิจารณาปัจจัยอิสระ 
# ได้แก่ อายุ ชั่วโมงการเรียนรู้ ระดับการศึกษาของผู้เล่น และแรงจูงใจจากเพื่อน
# ตัวแปรตาม (Dependent Variable):

# Hours per Day: จำนวนชั่วโมงที่เล่นเกม ROV ต่อวัน
# ตัวแปรอิสระ (Independent Variables):

# Age: อายุของผู้เล่น (ปี)
# Study Hours: ชั่วโมงการเรียนรู้
# Education Level: ระดับการศึกษา (1 = ประถมศึกษา, 2 = มัธยมศึกษา, 3 = ปริญญาตรี, 4 = สูงกว่าปริญญาตรี)
# Peer Motivation (1-10): คะแนนแรงจูงใจจากเพื่อนให้เล่นเกม ROV (1 = น้อยที่สุด, 10 = มากที่สุด)
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# กำหนดข้อมูล DataFrame (n=30)
data = {
    "Age": [18, 20, 25, 22, 30, 19, 28, 24, 21, 26,
            29, 22, 23, 27, 19, 20, 24, 22, 25, 21,
            18, 30, 28, 26, 29, 23, 22, 19, 24, 20],
    "Study Hours": [4.0, 3.5, 2.0, 4.5, 1.0, 3.8, 1.5, 3.2, 4.0, 2.5,
                    1.8, 3.6, 2.9, 1.2, 4.3, 4.1, 3.7, 4.0, 2.0, 3.4,
                    3.9, 1.0, 1.5, 2.2, 1.8, 3.5, 4.2, 4.0, 3.3, 3.8],
    "Education Level": [2, 3, 3, 2, 4, 2, 3, 2, 2, 3,
                        4, 2, 3, 3, 2, 2, 2, 3, 3, 2,
                        2, 4, 3, 3, 4, 3, 3, 2, 3, 2],
    "Peer Motivation": [8, 9, 7, 8, 6, 9, 7, 8, 8, 7,
                        6, 8, 8, 7, 9, 9, 8, 8, 7, 8,
                        9, 6, 7, 7, 6, 8, 8, 9, 8, 9],
    "Hours per Day": [3.0, 2.5, 4.0, 3.2, 5.0, 3.0, 4.5, 3.5, 3.6, 4.2,
                      4.8, 3.7, 3.9, 4.3, 2.8, 2.9, 3.8, 3.4, 4.2, 3.1,
                      2.7, 5.0, 4.6, 4.0, 4.9, 3.5, 3.2, 2.8, 3.6, 3.0]
}

df = pd.DataFrame(data)

# ตัวแปรอิสระ (Age, Study Hours, Education Level, Peer Motivation)
X = df[["Age", "Study Hours", "Education Level", "Peer Motivation"]]

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
study_coef = model.params['Study Hours']
education_coef = model.params['Education Level']
peer_motivation_coef = model.params['Peer Motivation']

# สร้างกราฟเปรียบเทียบ
plt.figure(figsize=(12, 6))
plt.scatter(y, y_pred, color="blue", label="Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linewidth=2, label="Ideal Fit")
plt.title("Actual vs Predicted ROV Playing Hours (Hours per Day)")
plt.xlabel("Actual Playing Hours (Hours)")
plt.ylabel("Predicted Playing Hours (Hours)")
plt.legend()
plt.grid()
plt.show()

# พิมพ์สมการของโมเดล
print("\nสมการของโมเดล:")
print(f"Hours per Day = {intercept:.2f} + ({age_coef:.2f} * Age) + ({study_coef:.2f} * Study Hours) + "
      f"({education_coef:.2f} * Education Level) + ({peer_motivation_coef:.2f} * Peer Motivation)")

# แสดงค่า R-squared
print(f"\nR-squared: {r2:.2f}")

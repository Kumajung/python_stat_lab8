import pandas as pd

# โหลดไฟล์ Excel
file_path = "Coffee.xlsx"  # เปลี่ยนเป็นพาธไฟล์ของคุณ
xls = pd.ExcelFile(file_path)

# โหลดข้อมูลจากชีต "Export_Output" โดยไม่ใช้ header อัตโนมัติ
df_raw = pd.read_excel(xls, sheet_name="Export_Output", header=None)

# กำหนดแถวแรกเป็นหัวตาราง (column names)
headers = df_raw.iloc[0]  # ดึงแถวแรกเป็น header
df = df_raw[1:].reset_index(drop=True)  # ลบแถวแรกออกจากข้อมูลจริง
df.columns = headers  # ตั้งชื่อคอลัมน์ใหม่

# แยกข้อมูลโดยใช้ comma (",") เป็นตัวแบ่ง (เฉพาะแถวข้อมูล)
df_split = df[df.columns[0]].str.split(",", expand=True)

# ตั้งชื่อคอลัมน์ใหม่โดยใช้ header ที่แยกมา
df_split.columns = headers.str.split(",", expand=True).iloc[0]

# บันทึกไฟล์ใหม่เป็น Excel
output_path = "Coffee_Split.xlsx"
df_split.to_excel(output_path, index=False)

print(f"ไฟล์ที่แยกข้อมูลถูกบันทึกเป็น: {output_path}")

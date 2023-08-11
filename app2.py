import streamlit as st
import cv2
import numpy as np
from lib_detection import detect_lp, im2single, load_model
import mysql.connector
import glob
from datetime import date
import os

# Define the function to sort contours from left to right
def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# Define the list of valid characters on license plates
char_list = '0123456789ABCDEFGHIKLMNPRSTUVXYZ'

# Define the function to fine-tune license plate characters
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

# Load LP detection model
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Title of the Streamlit app
st.title("Number Plate Recognition and Parking Management")

# File Upload widget
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    # Find the most recent file
    folder_path = r"D:\BigdataStream\Project_NumberPlate\takepicture"
    file_type = r'\*jpg'
    files = glob.glob(folder_path + file_type)
    if not files:
        st.write('Folder Empty!')
    else:
        max_file = max(files, key=os.path.getctime)

    Dmax = 608
    Dmin = 288
    ratio = float(max(image.shape[:2])) / min(image.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _, LpImg, lp_type = detect_lp(wpod_net, im2single(image), bound_dim, lp_threshold=0.5)

    digit_w = 30
    digit_h = 60
    model_svm = cv2.ml.SVM_load('svm.xml')

    if len(LpImg):
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        roi = LpImg[0]
        gray = cv2.cvtColor(LpImg[0], cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _ = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        plate_info = ""

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1.5 <= ratio <= 3.5:
                if h / roi.shape[0] >= 0.6:
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    curr_num = thre_mor[y:y+h, x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num, dtype=np.float32)
                    curr_num = curr_num.reshape(-1, digit_w * digit_h)
                    result = model_svm.predict(curr_num)[1]
                    result = int(result[0, 0])

                    if result <= 9:
                        result = str(result)
                    else:
                        result = chr(result)

                    plate_info += result

        # Kiểm tra xem biển số có hợp lệ là biển số xe máy không
        valid_motorbike_plate = False
        if len(plate_info) >= 5:
            if plate_info[2] == '-' or plate_info[4] == '-':
                valid_motorbike_plate = True

        if valid_motorbike_plate:
            # Hiển thị ảnh đã xử lý
            st.image(roi, caption="Number Plate Detected", use_column_width=True)

            # Hiển thị nút Lưu Xe Máy
            save_motorbike_button = st.button("Lưu Xe Máy")
            if save_motorbike_button:
                # Thực hiện lưu thông tin xe máy vào cơ sở dữ liệu
                mydb = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="ohhhchank3",
                    database="fpt23_team3"
                )

                mycursorInsert = mydb.cursor()
                datein = date.today()
                formatted_date = datein.strftime('%Y-%m-%d')
                queryInsert = "INSERT INTO info_vehicle_in (number_plate, time_in) VALUES (%s, %s)"
                valInsert = (plate_info, formatted_date)
                mycursorInsert.execute(queryInsert, valInsert)
                mydb.commit()
                st.write(mycursorInsert.rowcount, "details inserted")
                mydb.close()

        else:
            # Hiển thị ảnh đã xử lý
            st.image(roi, caption="Number Plate Detected", use_column_width=True)

            # Hiển thị nút Lưu Xe Ô Tô
            save_car_button = st.button("Lưu Xe Ô Tô")
            if save_car_button:
                # Thực hiện lưu thông tin xe ô tô vào cơ sở dữ liệu
                mydb = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="ohhhchank3",
                    database="fpt23_team3"
                )

                mycursorInsert = mydb.cursor()
                datein = date.today()
                formatted_date = datein.strftime('%Y-%m-%d')
                queryInsert = "INSERT INTO info_vehicle_in (number_plate, time_in) VALUES (%s, %s)"
                valInsert = (plate_info, formatted_date)
                mycursorInsert.execute(queryInsert, valInsert)
                mydb.commit()
                st.write(mycursorInsert.rowcount, "details inserted")
                mydb.close()

# ... (phần mã xử lý và hiển thị thông tin tương tự như phần trước)

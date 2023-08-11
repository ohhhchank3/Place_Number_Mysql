import cv2
import pytesseract

def preprocess_image(image):
    # Chuyển ảnh thành ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng bộ lọc Gauss để làm mờ ảnh
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Áp dụng phép biến đổi Canny để phát hiện biên cạnh
    edges = cv2.Canny(blurred, 50, 150)

    return edges

def extract_license_plate(image):
    # Tiền xử lý ảnh
    preprocessed_image = preprocess_image(image)

    # Tìm các contour trên ảnh
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các contour theo diện tích giảm dần
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        # Xác định hình chữ nhật bao quanh contour
        x, y, w, h = cv2.boundingRect(contour)

        # Kiểm tra tỷ lệ chiều dài và chiều rộng của hình chữ nhật
        aspect_ratio = w / float(h)
        if 1.5 <= aspect_ratio <= 4.0:
            # Trích xuất ảnh biển số xe
            license_plate = image[y:y+h, x:x+w]

            return license_plate

    return None

def recognize_license_plate(image):
    # Chuyển đổi ảnh sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng phép nhị phân hóa để tăng độ tương phản
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Sử dụng PyTesseract để nhận dạng kí tự
    config = r'--oem 3 --psm 7'
    plate_info = pytesseract.image_to_string(binary, config=config)

    return plate_info

def process_license_plate(image):
    # Xử lý hàng đầu tiên
    first_row = extract_license_plate(image)
    if first_row is not None:
        plate_info_first_row = recognize_license_plate(first_row)
    else:
        plate_info_first_row = ""

    # Xử lý hàng thứ hai
    second_row = extract_license_plate(image[int(image.shape[0]/2):])
    if second_row is not None:
        plate_info_second_row = recognize_license_plate(second_row)
    else:
        plate_info_second_row = ""

    # Kết hợp kết quả từ hai hàng
    plate_info = plate_info_first_row + plate_info_second_row

    return plate_info

# Load ảnh
image = cv2.imread(r"D:/BigdataStream/Project_NumberPlate/takepicture/test.jpg")

# Xử lý biển số xe
plate_info = process_license_plate(image)

# In kết quả
print("Biển số xe: ", plate_info)
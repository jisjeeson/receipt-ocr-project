import re
import cv2
import numpy as np
from paddleocr import PaddleOCR

# ---------------------------------
# INITIALIZE OCR
# ---------------------------------
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

# ---------------------------------
# IMAGE PREPROCESSING
# ---------------------------------
def resize_large_image(img_path):

    image = cv2.imread(img_path)

    if image is None:
        return img_path

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal
    denoised = cv2.fastNlMeansDenoising(gray)

    # Sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Adaptive threshold
    processed = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Resize image for better OCR
    processed = cv2.resize(
        processed,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    output_path = "processed_receipt.png"

    cv2.imwrite(output_path, processed)

    return output_path


# ---------------------------------
# COMPANY EXTRACTION
# ---------------------------------
def extract_company(text):

    lines = text.split("\n")

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if len(line) > 3:
            cleaned_lines.append(line)

    keywords = [
        "SDN",
        "BHD",
        "RESTAURANT",
        "STORE",
        "SHOP",
        "CAFE",
        "MART",
        "TRADING",
        "ENTERPRISE",
        "FOOD",
        "HOTEL",
        "BAKERY"
    ]

    for line in cleaned_lines[:12]:

        upper_line = line.upper()

        if any(keyword in upper_line for keyword in keywords):
            return upper_line

    if cleaned_lines:
        return cleaned_lines[0].upper()

    return "Not Found"


# ---------------------------------
# DATE EXTRACTION
# ---------------------------------
def extract_date(text):

    patterns = [

        r"\d{2}/\d{2}/\d{4}",

        r"\d{2}-\d{2}-\d{4}",

        r"\d{4}-\d{2}-\d{2}",

        r"\d{2}/\d{2}/\d{2}",

        r"\d{2}-\d{2}-\d{2}"
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            return match.group()

    return "Not Found"


# ---------------------------------
# TOTAL EXTRACTION
# ---------------------------------
def extract_total(text):

    text_upper = text.upper()

    patterns = [

        r"GRAND TOTAL\s*[:\-]?\s*RM?\s*([0-9]+\.[0-9]{2})",

        r"TOTAL\s*[:\-]?\s*RM?\s*([0-9]+\.[0-9]{2})",

        r"NETT TOTAL\s*[:\-]?\s*RM?\s*([0-9]+\.[0-9]{2})",

        r"AMOUNT\s*[:\-]?\s*RM?\s*([0-9]+\.[0-9]{2})",

        r"CASH\s*RM?\s*([0-9]+\.[0-9]{2})"
    ]

    for pattern in patterns:

        match = re.search(pattern, text_upper)

        if match:
            return match.group(1)

    # Fallback -> largest amount
    amounts = re.findall(r"\d+\.\d{2}", text_upper)

    if amounts:

        amounts = [float(x) for x in amounts]

        return str(max(amounts))

    return "Not Found"


# ---------------------------------
# MAIN OCR PROCESS
# ---------------------------------
def process_receipt(img_path):

    # Preprocess image
    processed_img = resize_large_image(img_path)

    # OCR
    result = ocr.ocr(processed_img, cls=True)

    text = ""

    if result and result[0]:

        for line in result[0]:

            detected_text = line[1][0]

            text += detected_text + "\n"

    # Extract fields
    extracted = {

        "company": extract_company(text),

        "date": extract_date(text),

        "total": extract_total(text)
    }

    return extracted
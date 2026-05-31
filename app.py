import streamlit as st
from PIL import Image
import pandas as pd
from utils import process_receipt

st.set_page_config(page_title="Receipt OCR System")

st.title("🧾 Receipt OCR Extraction System")

uploaded_file = st.file_uploader(
    "Upload Receipt Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Receipt", use_container_width=True)

    with open("temp_receipt.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    result = process_receipt("temp_receipt.png")
    st.write(result)

    st.subheader("Extracted Information")

    df = pd.DataFrame([result])

    st.table(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        "receipt_data.csv",
        "text/csv"
    )
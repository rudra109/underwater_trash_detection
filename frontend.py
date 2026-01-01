import streamlit as st
import requests
from PIL import Image
import pandas as pd
import os

#  PAGE CONFIG 
st.set_page_config(
    page_title="Water Pollution Detection AI",
    page_icon="*",
    layout="centered"
)

st.title(" Water Pollution Detection AI")
st.write(
    "Upload an underwater image to detect pollution and generate "
    "structured environmental records."
)

#  IMAGE UPLOAD 
uploaded_file = st.file_uploader(
    "Upload Underwater Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

#  LOCATION & DEPTH  (still under construction) manually entering longitutde and latitude and depth 
st.subheader("📍 Detection Metadata")

latitude = st.number_input("Latitude", value=15.4912, format="%.6f")
longitude = st.number_input("Longitude", value=73.8185, format="%.6f")
depth = st.number_input("Depth (meters)", value=0.0, step=0.1)

#  DETECT BUTTON 
if st.button("🚀 Detect Pollution"):

    if not uploaded_file:
        st.warning("Please upload an image first.")
    else:
        with st.spinner("Running AI detection..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(
                "http://127.0.0.1:8000/detect/",
                files=files,
                data={
                    "latitude": latitude,
                    "longitude": longitude,
                    "depth": depth
                }
            )

            if response.status_code == 200:
                data = response.json()
                records = data.get("detections", [])

                st.subheader("📋 Pollution Detection Records")

                if len(records) == 0:
                    st.success("No pollution detected 🌊")
                else:
                    # table
                    df = pd.DataFrame(records)
                    st.dataframe(df, use_container_width=True)

                    # detail
                    st.subheader("🧠 Detection Details")

                    for r in records:
                        st.markdown("---")
                        st.write(f"🕒 **Datetime:** {r['datetime']}")
                        st.write(
                            f"📍 **Location:** ({r['lat']}, {r['lon']}) | "
                            f"🌊 **Depth:** {r['depth']} m"
                        )
                        st.write(f"🗑️ **Class:** {r['class']}")
                        st.write(f"📊 **Confidence:** {r['confidence'] * 100:.2f}%")

                        # Show cropped detected object
                        if r["image_clip"] and os.path.exists(r["image_clip"]):
                            st.image(
                                r["image_clip"],
                                caption="Detected Object (Image Clip)",
                                width=250
                            )

            else:
                st.error("❌ Error connecting to detection server")

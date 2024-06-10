import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Fungsi untuk ekstraksi warna dominan
def extract_dominant_colors(image, k=5):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors = colors[sorted_indices]
    return dominant_colors

# Judul aplikasi
st.title('Dominant Color Picker')

# Unggah gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Tampilkan gambar yang diunggah
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Ekstraksi warna dominan
    st.write("Extracting dominant colors...")
    dominant_colors = extract_dominant_colors(image, k=5)

    # Tampilkan warna dominan
    st.write("Dominant colors:")
    for color in dominant_colors:
        st.write(f"RGB: {color}")
        st.markdown(
            f'<div style="background-color: rgb({color[0]}, {color[1]}, {color[2]}); width: 100px; height: 100px;"></div>',
            unsafe_allow_html=True
        )

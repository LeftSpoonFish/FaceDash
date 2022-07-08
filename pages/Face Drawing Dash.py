"""Application to test image annotations.

Run using: streamlit run Face Drawing Dash.py

@Author Slovenm
"""

import streamlit as st
import pathlib
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import io
import base64
from PIL import Image

STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

# We create a downloads directory within the streamlit static asset directory
# and we write output files to it.
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()


def get_image_download_link(img, filename, text):
    """Generates a link to download a particular image file."""
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


# Set title.
st.sidebar.title('Image Annotation')


# Specify canvas parameters in application
uploaded_file = st.sidebar.file_uploader("Upload Image to annotate:", type=["png", "jpg"])
image = None
res = None
if uploaded_file is None:
    st.write('Select an image to annotate')

if uploaded_file is not None:

    # Convert the file to an opencv image.
    st.image(uploaded_file, caption='Original Image')
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    # Display a selection box for choosing the filter to apply.
    option = st.sidebar.selectbox('Activate Annotate?', ('None', 'Image Annotate'))

    if option == 'Instructions':
        st.write('Choose an image to annotate') #Mayhaps putting an image gallery here to select which image from previously processed??

    elif option == 'Image Annotate':

        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        h, w = image.shape[:2]
        if w > 800:
            h_, w_ = int(h * 800 / w), 800
        else:
            h_, w_ = h, w

        # Create a canvas component.
        

# Specify canvas annotation parameters in application
        drawing_mode = st.sidebar.radio(
            "Drawing tool:",
            ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
        )
        #stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)


        if drawing_mode == 'point':
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        #stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        bg_color = None
        
        realtime_update = st.sidebar.checkbox("Update in realtime", True)

#Original Canvas drawing freeform
        #fill_color=st.sidebar.color_picker('Fill Color', disabled=False)
        #alphaT=st.sidebar.slider("Alpha", 1, 100, 2)
        canvas_result = st_canvas(
            fill_color="rgba(0,128,128,0)", #This is what controls the shapes background fill 
            stroke_width=stroke_width,
            stroke_color=st.sidebar.color_picker('Stroke Color',disabled=False),
            background_image=Image.open(uploaded_file).resize((h_, w_)),
            update_streamlit=True,
            height=h_,
            width=w_,
            drawing_mode=drawing_mode,
            key="canvas",
        )
        stroke = canvas_result.image_data

        if stroke is not None:

            if st.sidebar.checkbox('show mask'):
                st.image(stroke)

            mask = cv2.split(stroke)[3]
            mask = np.uint8(mask)
            mask = cv2.resize(mask, (w, h))

        st.sidebar.caption('Happy with the selection?')
        option = st.sidebar.selectbox('Mode', ['None', 'Telea', 'NS', 'Compare both'])

        if option == 'Telea':
            st.subheader('Result of Telea')
            res = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)[:,:,::-1]
            st.image(res)
        elif option == 'Compare both':
            col1, col2 = st.columns(2)
            res1 = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)[:,:,::-1]
            res2 = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_NS)[:,:,::-1]
            with col1:
                st.subheader('Result of Telea')
                st.image(res1)
            with col2:
                st.subheader('Result of NS')
                st.image(res2)
            if res1 is not None:
                # Display link.
                result1 = Image.fromarray(res1)
                st.sidebar.markdown(
                    get_image_download_link(result1, 'telea.png', 'Download Output of Telea'),
                    unsafe_allow_html=True)
            if res2 is not None:
                # Display link.
                result2 = Image.fromarray(res2)
                st.sidebar.markdown(
                    get_image_download_link(result2, 'ns.png', 'Download Output of NS'),
                    unsafe_allow_html=True)

        else:
            pass

    if res is not None:
        # Display link.
        result = Image.fromarray(res)
        st.sidebar.markdown(
            get_image_download_link(result, 'output.png', 'Download Output'),
            unsafe_allow_html=True)

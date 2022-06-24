#Modified by Mark 2022 to contain 3D facial code
#3D Face Landmark User Interface with StreamLit

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.image import thumbnail
import plotly.graph_objects as go
import plotly.express as px

import base64 #for GIF animation



#animated GIF
file_gif=open("giftest.gif","rb")
contents=file_gif.read()
data_url=base64.b64encode(contents).decode("utf-8")
file_gif.close()
#st.image('giftest.gif')
#st.markdown(f'<img src="data:image/gif;base64,{data_url}" width="300" height="200" alt="giftest.gif" class="center">,unsafe_allow_html=True,)


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

#DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = ('/Users/mark/Bio/Faces/logo.jpg')

st.title('Face Mesh Application using MediaPipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Face Mesh Application using MediaPipe')
st.sidebar.subheader('Parameters')

@st.cache()
def image_resize(image, width=.5, height=.5, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

app_mode = st.sidebar.selectbox('Navigate:',
['Introduction','Process an Image']
)

if app_mode =='Introduction':
    st.image('giftest.gif')
    st.markdown('In this application we are using **MediaPipe** to create a Face Mesh. **StreamLit** is used to create the Web Graphical User Interface (GUI) ')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )


    st.markdown('''
          # About: \n 
            This is ** Mark ** from **LSF**. \n
           
            This app creates 3D renders of 2D faces
        
            If you are feeling generous you can buy me a **cup of  coffee ** from [HERE](www.LeftSpoonFish.com)
             
            ''')


elif app_mode =='Process an Image':

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    st.sidebar.markdown('---')

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,)
    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")
    st.markdown('---')

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        st.sidebar.write('Nothing to process yet')
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

#    st.sidebar.text('Original Image')
   
    face_count = 0
    # Dashboard
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            #print('face_landmarks:', face_landmarks)

            mp_drawing.draw_landmarks(
            image=out_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Original Image')
        st.image(image,use_column_width=False)
        st.subheader('Facial Mesh')
        st.image(out_image,use_column_width= False)

    #Additional Testing
    st.write('---')
    st.subheader('3D Process')
    keypoints = []
    for data_point in face_landmarks.landmark:
        keypoints.append({
            'X': data_point.x,
            'Y': data_point.y,
            'Z': data_point.z,
            'Visibility': data_point.visibility,
        })
    xlist = [x['X'] for x in keypoints]
    ylist = [y['Y'] for y in keypoints]
    zlist = [z['Z'] for z in keypoints]
        
    df = pd.DataFrame(columns=['x','y','z'])
    df.x = xlist
    df.y = ylist
    df.z = zlist
    st.write('Table of Coordinates')
    st.dataframe(df)

    #3D Scatter plot
    fig = px.scatter_3d(df, x='x',y='y',z='z')

    fig.update_layout(plot_bgcolor='rgb(12,163,135)',paper_bgcolor='rgb(0,0,0)',#coloraxis={"colorbar": {"x": -0.2, "len": 0.5, "y": 0.8}}, #I think this is for contours
                      scene = dict(
                          xaxis = dict(
                              backgroundcolor="rgba(0, 0, 0,0)",
                              gridcolor="black",
                              showbackground=False,
                              zerolinecolor="black",
                              showticklabels=False),
                          yaxis = dict(
                              backgroundcolor="rgba(0, 0, 0,0)",
                              gridcolor="black",
                              showbackground=False,
                              zerolinecolor="black",
                              showticklabels=False),
                          zaxis = dict(
                              backgroundcolor="rgba(0, 0, 0,0)",
                              gridcolor="black",
                              showbackground=False,
                              zerolinecolor="black",
                              showticklabels=False),),
                         )

    fig.update_traces(marker=dict(size=1.5, color='orange'))
    st.plotly_chart(fig, use_container_width=True)
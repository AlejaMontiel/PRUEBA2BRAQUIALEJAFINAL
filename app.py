import os
import io
import zipfile
import tempfile
import random

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import SimpleITK as sitk
from skimage.transform import resize
import plotly.graph_objects as go
import pandas as pd

# Configuración de la página
st.set_page_config(layout="wide", page_title="BrachyCervix")

# Estilos globales
def inject_css():
    st.markdown("""
    <style>
        .giant-title { color: #28aec5; text-align: center; font-size: 72px; margin: 30px 0; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
        .sub-header { color: #c0d711; font-size: 24px; margin-bottom: 15px; font-weight: bold; }
        .stButton>button { background-color: #28aec5; color: white; border: none; border-radius: 4px; padding: 8px 16px; }
        .stButton>button:hover { background-color: #1c94aa; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# Sidebar: carga y configuración
st.sidebar.markdown('<p class="sub-header">Visualizador DICOM</p>', unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("ZIP archivos DICOM", type="zip")

@st.cache_data
def load_dicom_series(zip_bytes):
    temp = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as z:
        z.extractall(temp)
    series = []
    for root, _, _ in os.walk(temp):
        try:
            ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
            for sid in ids:
                files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, sid)
                if files:
                    series.append((sid, files))
        except:
            pass
    return series

# Carga DICOM
dicom_series = []
if uploaded:
    dicom_series = load_dicom_series(uploaded.read())
    if dicom_series:
        choices = [f"Serie {i+1}: {sid[:10]} ({len(files)} cortes)" for i, (sid, files) in enumerate(dicom_series)]
        sel = st.sidebar.selectbox("Selecciona serie:", choices)
        idx = choices.index(sel)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_series[idx][1])
        vol = reader.Execute()
        img = sitk.GetArrayViewFromImage(vol)
        original = img.copy()
    else:
        st.sidebar.error("No DICOM válidos en ZIP")
else:
    img = None

# Función de ventana
def apply_window(image, ww, wc):
    arr = image.astype(float)
    mn, mx = wc - ww/2, wc + ww/2
    return np.clip((arr - mn) / (mx - mn), 0, 1)

# Mostrar imágenes 2D
if img is not None:
    n_ax, n_cor, n_sag = img.shape
    mn, mx = float(img.min()), float(img.max())
    default = {'ww': mx-mn, 'wc': mn + (mx-mn)/2}

    sync = st.sidebar.checkbox('Sincronizar cortes', True)
    if sync:
        orientation = st.sidebar.radio('Corte', ['Axial','Coronal','Sagital'])
        limits = {'Axial':n_ax,'Coronal':n_cor,'Sagital':n_sag}
        idx = st.sidebar.slider('Índice', 0, limits[orientation]-1, limits[orientation]//2)
    else:
        orientation = st.sidebar.selectbox('Corte', ['Axial','Coronal','Sagital'])
        idx = st.sidebar.slider('Índice', 0, img.shape[['Axial','Coronal','Sagital'].index(orientation)]-1,
                                 img.shape[['Axial','Coronal','Sagital'].index(orientation)]//2)
    invert = st.sidebar.checkbox('Negativo', False)
    wtype = st.sidebar.selectbox('Tipo ventana', ['Default','Abdomen','Hueso','Pulmón'])
    presets = {'Abdomen':(400,40),'Hueso':(2000,500),'Pulmón':(1500,-600)}
    ww, wc = presets.get(wtype, (default['ww'], default['wc']))

    slices = {
        'Axial': img[idx,:,:],
        'Coronal': img[:,idx,:],
        'Sagital': img[:,:,idx]
    }
    cols = st.columns(3)
    for col,(name,data) in zip(cols, slices.items()):
        with col:
            st.markdown(name)
            fig, ax = plt.subplots(); ax.axis('off')
            img2d = apply_window(data, ww, wc)
            if invert: img2d = 1 - img2d
            ax.imshow(img2d, cmap='gray', origin='lower'); st.pyplot(fig)

    # 3D y agujas
    if st.sidebar.checkbox('Mostrar 3D', True):
        resized = resize(original, (64,64,64), anti_aliasing=True)
        if 'needles' not in st.session_state:
            st.session_state['needles'] = []

        # Controles de creación con cantidad múltiple
        with st.expander('Nueva aguja'):
            mode = st.radio('Modo', ['Manual','Aleatoria'], horizontal=True)
            shape = st.radio('Forma', ['Recta','Curva'], horizontal=True)
            count = st.number_input('Cantidad aleatoria', min_value=1, value=1, step=1)
            if mode == 'Manual':
                c1, c2 = st.columns(2)
                with c1:
                    x1 = st.number_input('X1', 0.0, 64.0, 32.0)
                    y1 = st.number_input('Y1', 0.0, 64.0, 32.0)
                    z1 = st.number_input('Z1', 0.0, 64.0, 32.0)
                with c2:
                    x2 = st.number_input('X2', 0.0, 64.0, 32.0)
                    y2 = st.number_input('Y2', 0.0, 64.0, 32.0)
                    z2 = st.number_input('Z2', 0.0, 64.0, 32.0)
            if st.button('Agregar aguja'):
                # Generar una o varias según modo
                times = count if mode == 'Aleatoria' else 1
                for _ in range(times):
                    if mode == 'Aleatoria':
                        xa,ya,za = [random.uniform(7,35) for _ in range(3)]
                        xb,yb,zb = [random.uniform(30,45) for _ in range(3)]
                    pts = ((x1,y1,z1),(x2,y2,z2)) if mode == 'Manual' else ((xa,ya,za),(xb,yb,zb))
                    st.session_state['needles'].append({
                        'points': pts,
                        'color': f"#{random.randint(0,0xFFFFFF):06x}",
                        'curved': (shape == 'Curva')
                    })

        # Tabla editable
        st.markdown('### Registro de agujas')
        df = pd.DataFrame([{**{'ID':i+1,
                                'X1':round(p[0],1),'Y1':round(p[1],1),'Z1':round(p[2],1),
                                'X2':round(q[0],1),'Y2':round(q[1],1),'Z2':round(q[2],1),
                                'Color':d['color'],'Forma':('Curva' if d['curved'] else 'Recta'),'Eliminar':False}}
                             for i,d in enumerate(st.session_state['needles'])
                             for p,q in [d['points']]])
        edited = st.data_editor(df, use_container_width=True)
        # Actualizar estado
        st.session_state['needles'] = []
        for _, r in edited.iterrows():
            if not r['Eliminar']:
                pts = ((r['X1'],r['Y1'],r['Z1']), (r['X2'],r['Y2'],r['Z2']))
                st.session_state['needles'].append({'points': pts, 'color': r['Color'], 'curved': (r['Forma']=='Curva')})

        # Render 3D
        xg, yg, zg = np.mgrid[0:64,0:64,0:64]
        fig3d = go.Figure(data=[go.Volume(
            x=xg.flatten(), y=yg.flatten(), z=zg.flatten(),
            value=resized.flatten(), opacity=0.1, surface_count=15, colorscale='Gray'
        )])
        for d in st.session_state['needles']:
            (x1,y1,z1),(x2,y2,z2) = d['points']
            if d['curved']:
                t = np.linspace(0,1,50);
                xs = x1*(1-t)+x2*t; ys = y1*(1-t)+y2*t;
                zs = z1*(1-t)+z2*t + 5*np.sin(np.pi*t)
            else:
                xs, ys, zs = [x1,x2], [y1,y2], [z1,z2]
            fig3d.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines+markers',
                marker=dict(size=4, color=d['color']),
                line=dict(width=3, color=d['color'])
            ))
        fig3d.update_layout(margin=dict(l=0,r=0,b=0,t=0))
        st.subheader('Vista 3D')
        st.plotly_chart(fig3d, use_container_width=True)

# Pie de página
st.markdown('<p class="giant-title">BrachyCervix</p>', unsafe_allow_html=True)
st.markdown("""
<hr>
<div style="text-align:center;color:#28aec5;font-size:50px;">
    BrachyCervix - Semiautomátización y visor para procesos de braquiterapia enfocados en el Cervix
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    Proyecto asignatura medialab 3
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    Universidad EAFIT
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    Clínica Las Américas AUNA
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    - Nicolás Ramirez
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    - Alejandra Montiel
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    - María Camila Díaz
</div>
<div style="text-align:center;color:#28aec5;font-size:20px;">
    - María Paula Jaimes
</div>
""", unsafe_allow_html=True)

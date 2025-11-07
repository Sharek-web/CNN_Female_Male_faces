import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import urllib.request

# Configuración de la página
st.set_page_config(
    page_title="Gender AI Classifier",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS inspirado en el diseño retro/vintage
st.markdown("""
    <style>
    /* Importar fuente retro */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');
    
    /* Fondo con colores vintage */
    .stApp {
        background: linear-gradient(135deg, #2d3436 0%, #1e272e 50%, #34495e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 4rem 2rem;
        position: relative;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 5rem;
        font-weight: 900;
        color: #f4e4c1;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        text-shadow: 3px 3px 0px rgba(0,0,0,0.3);
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #e8b86d;
        font-style: italic;
        margin-bottom: 2rem;
    }
    
    .hero-description {
        color: #d4c5a9;
        font-size: 1.1rem;
        max-width: 700px;
        margin: 0 auto 2rem auto;
        line-height: 1.8;
    }
    
    /* Decorative stars */
    .star {
        color: #f4e4c1;
        font-size: 2rem;
        display: inline-block;
        margin: 0 1rem;
        animation: twinkle 2s infinite;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    /* Menu desplegable */
    .menu-container {
        background: rgba(244, 228, 193, 0.1);
        border: 2px solid #e8b86d;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 2rem auto;
        max-width: 600px;
        backdrop-filter: blur(10px);
    }
    
    .menu-item {
        background: linear-gradient(135deg, #e8b86d 0%, #c99850 100%);
        color: #2d3436;
        padding: 1.2rem 2rem;
        margin: 1rem 0;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid #f4e4c1;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .menu-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(232, 184, 109, 0.4);
    }
    
    /* Botones personalizados */
    .stButton>button {
        background: linear-gradient(135deg, #e8b86d 0%, #c99850 100%);
        color: #2d3436;
        border: 2px solid #f4e4c1;
        border-radius: 12px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(232, 184, 109, 0.4);
    }
    
    /* Tarjetas de contenido */
    .content-card {
        background: rgba(244, 228, 193, 0.08);
        border: 2px solid rgba(232, 184, 109, 0.3);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .content-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        color: #f4e4c1;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .content-text {
        color: #d4c5a9;
        font-size: 1.1rem;
        line-height: 1.8;
        text-align: center;
    }
    
    /* Resultado de predicción */
    .prediction-box {
        background: linear-gradient(135deg, rgba(232, 184, 109, 0.2) 0%, rgba(201, 152, 80, 0.2) 100%);
        border: 3px solid #e8b86d;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .prediction-emoji {
        font-size: 6rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3));
    }
    
    .prediction-label {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: #f4e4c1;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .prediction-confidence {
        font-size: 1.8rem;
        color: #e8b86d;
        font-weight: 600;
    }
    
    /* Tabs personalizados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(244, 228, 193, 0.1);
        border-radius: 15px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: #d4c5a9;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e8b86d 0%, #c99850 100%);
        color: #2d3436;
        border: 2px solid #f4e4c1;
    }
    
    /* Divider decorativo */
    .divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #e8b86d, transparent);
        margin: 3rem 0;
        position: relative;
    }
    
    .divider::after {
        content: "✦";
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        background: #2d3436;
        padding: 0 1rem;
        color: #e8b86d;
        font-size: 1.5rem;
    }
    
    /* Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(244, 228, 193, 0.05);
        border: 2px dashed #e8b86d;
        border-radius: 16px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
        color: #d4c5a9;
    }
    
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #e8b86d 0%, #c99850 100%);
        color: #2d3436;
        border: none;
        font-weight: 600;
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(232, 184, 109, 0.15);
        border-left: 4px solid #e8b86d;
        color: #d4c5a9;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(244, 228, 193, 0.08);
        border: 2px solid rgba(232, 184, 109, 0.3);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e8b86d;
        margin: 1rem 0;
    }
    
    .metric-label {
        color: #d4c5a9;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)



WEIGHTS_URL = "https://drive.google.com/file/d/1cMB0PRX_7qkLIxw0AfMgzXrtAFyzc5jj/view?usp=drive_link"  # Ejemplo: "https://drive.google.com/uc?id=TU_ID&export=download"

# Función para descargar pesos desde URL
def download_weights(url, destination="model_weights.weights.h5"):
    """Descarga los pesos del modelo si no existen localmente"""
    if not os.path.exists(destination):
        try:
            st.info("📥 Descargando pesos del modelo...")
            urllib.request.urlretrieve(url, destination)
            st.success("✅ Pesos descargados correctamente")
        except Exception as e:
            st.error(f"❌ Error al descargar pesos: {str(e)}")
            return False
    return True

# Función para cargar el modelo
@st.cache_resource
def load_model():
    try:
        # Descargar pesos si es necesario
        if not download_weights(WEIGHTS_URL):
            st.error("⚠️ No se pudieron descargar los pesos del modelo")
            st.info("""
            **Para usar tu propio modelo:**
            1. Sube `model_weights.weights.h5` a Google Drive
            2. Haz clic derecho → "Obtener enlace" → "Cualquier persona con el enlace"
            3. Copia el ID del enlace (parte después de /d/ y antes de /view)
            4. Reemplaza la URL en el código con: 
               `https://drive.google.com/uc?id=TU_ID&export=download`
            """)
            return None
        
        # Recrear la arquitectura exacta del modelo
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(224,224,3)),
            BatchNormalization(),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2,2)),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2,2)),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        # Cargar los pesos
        model.load_weights("model_weights.weights.h5")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Función para preprocesar la imagen
def preprocess_image(image, target_size=(224, 224)):
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, target_size)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch, img_resized

# Función para generar Grad-CAM (CORREGIDA)
def generate_gradcam(model, img_array, layer_name=None):
    try:
        # Buscar la última capa convolucional
        if layer_name is None:
            for layer in reversed(model.layers):
                if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            st.warning("No se encontró una capa convolucional en el modelo")
            return None
        
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        
        return heatmap
    except Exception as e:
        st.error(f"Error en Grad-CAM: {str(e)}")
        return None

# Función para generar Saliency Map
def generate_saliency_map(model, img_array):
    try:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, img_tensor)
        grads = tf.math.abs(grads)
        saliency = tf.reduce_max(grads, axis=-1)[0]
        
        saliency = saliency.numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    except Exception as e:
        st.error(f"Error en Saliency Map: {str(e)}")
        return None

# Función para superponer heatmap
def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    return superimposed

# Inicializar estado de sesión
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'image' not in st.session_state:
    st.session_state.image = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# Cargar modelo
model = load_model()

if model is None:
    st.stop()

# ===== PÁGINA HOME =====
if st.session_state.page == 'home':
    st.markdown("""
    <div class="hero-section">
        <span class="star">✦</span>
        <h1 class="hero-title">GENDER<br/>CLASSIFIER</h1>
        <span class="star">✦</span>
        <p class="hero-subtitle">Un viaje hacia la inteligencia artificial</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="content-card">
            <h2 class="content-title">¿Qué es este modelo?</h2>
            <p class="content-text">
                Este sistema utiliza <strong>Redes Neuronales Convolucionales Profundas</strong> 
                para analizar rostros humanos y determinar características de género. 
                <br/><br/>
                Entrenado con miles de imágenes, el modelo ha aprendido a identificar 
                patrones sutiles en estructuras faciales, texturas y rasgos distintivos.
                <br/><br/>
                Pero no solo predice... también <strong>explica</strong> sus decisiones usando 
                técnicas avanzadas de interpretabilidad como <strong>Grad-CAM</strong> y 
                <strong>Saliency Maps</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        
        if st.button("🚀 Comenzar Análisis", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()

# ===== PÁGINA UPLOAD =====
elif st.session_state.page == 'upload':
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title" style="font-size: 3.5rem;">SUBE TU IMAGEN</h1>
        <p class="hero-subtitle" style="font-size: 1.5rem;">Comencemos el análisis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Arrastra una imagen aquí o haz clic para seleccionar",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.image = image
            
            st.markdown("<br/>", unsafe_allow_html=True)
            st.image(image, use_container_width=True, caption="Imagen cargada")
            
            st.markdown("<br/>", unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("⬅️ Volver", use_container_width=True):
                    st.session_state.page = 'home'
                    st.rerun()
            with col_b:
                if st.button("💻 Análizar", use_container_width=True):
                    with st.spinner('Analizando imagen...'):
                        img_preprocessed, img_resized = preprocess_image(image)
                        prediction = model.predict(img_preprocessed, verbose=0)[0][0]
                        
                        st.session_state.results = {
                            'prediction': prediction,
                            'img_preprocessed': img_preprocessed,
                            'img_resized': img_resized
                        }
                        st.session_state.prediction_done = True
                        st.session_state.page = 'results'
                        st.rerun()

# ===== PÁGINA RESULTADOS =====
elif st.session_state.page == 'results':
    prediction = st.session_state.results['prediction']
    
    if prediction > 0.5:
        gender = "Femenino"
        confidence = prediction * 100
        emoji = "👩"
    else:
        gender = "Masculino"
        confidence = (1 - prediction) * 100
        emoji = "👨"
    
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title" style="font-size: 3rem;">RESULTADO</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(st.session_state.image, use_container_width=True)
        
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-emoji">{emoji}</div>
            <div class="prediction-label">{gender}</div>
            <div class="prediction-confidence">{confidence:.1f}% de confianza</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Probabilidades
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 3rem;">👨</div>
                <div class="metric-value">{(1-prediction)*100:.1f}%</div>
                <div class="metric-label">Masculino</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 3rem;">👩</div>
                <div class="metric-value">{prediction*100:.1f}%</div>
                <div class="metric-label">Femenino</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="content-card">
            <h2 class="content-title" style="font-size: 2rem;">¿Cómo llegó a esta conclusión?</h2>
            <p class="content-text">
                El modelo analizó características específicas de la imagen para tomar su decisión.
                <br/><br/>
                ¿Quieres ver <strong>exactamente qué partes de la imagen</strong> influyeron en el resultado?
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br/>", unsafe_allow_html=True)
        
        col_x, col_y = st.columns(2)
        with col_x:
            if st.button("⬅️ Nueva Imagen", use_container_width=True):
                st.session_state.page = 'upload'
                st.session_state.prediction_done = False
                st.rerun()
        with col_y:
            if st.button("Explicación", use_container_width=True):
                st.session_state.page = 'interpretability'
                st.rerun()

# ===== PÁGINA INTERPRETABILIDAD =====
elif st.session_state.page == 'interpretability':
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title" style="font-size: 3rem;">INTERPRETABILIDAD</h1>
        <p class="hero-subtitle" style="font-size: 1.3rem;">Así piensa el modelo</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner('Generando mapas de interpretabilidad...'):
        img_preprocessed = st.session_state.results['img_preprocessed']
        img_resized = st.session_state.results['img_resized']
        
        gradcam_heatmap = generate_gradcam(model, img_preprocessed)
        saliency_map = generate_saliency_map(model, img_preprocessed)
    
    st.info("💡 Estos mapas visualizan qué regiones de la imagen influyeron más en la decisión del modelo")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Original", 
        "Grad-CAM Superpuesto", 
        "Grad-CAM Solo",
        "Saliency Map"
    ])
    
    with tab1:
        st.markdown("<br/>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img_resized, use_container_width=True, caption="Imagen Original")
    
    with tab2:
        if gradcam_heatmap is not None:
            gradcam_overlay = overlay_heatmap(img_resized, gradcam_heatmap, alpha=0.5)
            st.markdown("<br/>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(gradcam_overlay, use_container_width=True, caption="Grad-CAM Superpuesto")
            st.markdown("""
            <div class="content-text">
                🔴 <strong>Áreas rojas/amarillas:</strong> Mayor influencia en la decisión<br/>
                🔵 <strong>Áreas azules/verdes:</strong> Menor influencia
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        if gradcam_heatmap is not None:
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * gradcam_heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            heatmap_resized = cv2.resize(heatmap_colored, (img_resized.shape[1], img_resized.shape[0]))
            
            st.markdown("<br/>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(heatmap_resized, use_container_width=True, caption="Grad-CAM Puro")
            st.markdown("""
            <div class="content-text">
                Este mapa muestra las <strong>activaciones</strong> de la última capa convolucional
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        if saliency_map is not None:
            col_left, col_center, col_right = st.columns([1, 1, 1])
            
            with col_left:
                st.markdown("<br/>", unsafe_allow_html=True)
                saliency_overlay = overlay_heatmap(img_resized, saliency_map, alpha=0.5)
                st.image(saliency_overlay, use_container_width=True, caption="Saliency Superpuesto")
            
            with col_center:
                st.markdown("<br/>", unsafe_allow_html=True)
                saliency_colored = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
                saliency_colored = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)
                saliency_resized = cv2.resize(saliency_colored, (img_resized.shape[1], img_resized.shape[0]))
                st.image(saliency_resized, use_container_width=True, caption="Saliency Puro")
            
            with col_right:
                st.markdown("<br/>", unsafe_allow_html=True)
                st.image(img_resized, use_container_width=True, caption="Original")
            
            st.markdown("""
            <div class="content-text">
                <strong>Saliency Map:</strong> Muestra qué píxeles, si se modificaran, 
                causarían el mayor cambio en la predicción
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🏠 Inicio", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
    with col2:
        if st.button("⬅️ Ver Resultado", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
    with col3:
        if st.button("📸 Nueva Imagen", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()

# Footer
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #d4c5a9; padding: 2rem; font-size: 0.9rem;">
    <span class="star" style="font-size: 1rem;">✦</span>
    Desarrollado con Streamlit & TensorFlow
    <span class="star" style="font-size: 1rem;">✦</span>
</div>
""", unsafe_allow_html=True)
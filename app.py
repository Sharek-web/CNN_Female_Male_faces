import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
from io import BytesIO

# -------------------------------------------------------------
# CONFIGURACIÓN INICIAL DE LA APP
# -------------------------------------------------------------
st.set_page_config(page_title="Clasificador CNN", layout="centered")
st.title("🧠 Clasificación de Imágenes con CNN")
st.write("Sube una imagen y observa la predicción del modelo y sus mapas de interpretabilidad.")

# -------------------------------------------------------------
# CARGAR MODELO
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model

model = load_model()
st.success("Modelo cargado correctamente ✅")

# -------------------------------------------------------------
# FUNCIONES AUXILIARES
# -------------------------------------------------------------
def preprocess_image(img, target_size=(128, 128)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

def display_gradcam(original_img, heatmap, alpha=0.4):
    img = np.array(original_img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img

# -------------------------------------------------------------
# INTERFAZ DE CARGA DE IMAGEN
# -------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Sube una imagen (jpg o png):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    from PIL import Image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼 Imagen cargada", use_column_width=True)

    # ---------------------------------------------------------
    # PREPROCESAMIENTO
    # ---------------------------------------------------------
    img_array = preprocess_image(img)
    preds = model.predict(img_array)[0]
    
    # ---------------------------------------------------------
    # MOSTRAR RESULTADOS
    # ---------------------------------------------------------
    classes = ["Male", "Female"]  # ajusta según tus clases
    result = classes[np.argmax(preds)]
    st.subheader(f"Predicción: **{result}**")
    st.write(f"Probabilidades: {dict(zip(classes, preds.round(3)))}")

    # ---------------------------------------------------------
    # MAPAS DE INTERPRETABILIDAD (Grad-CAM)
    # ---------------------------------------------------------
    st.subheader("🧩 Mapa Grad-CAM")

    # Ajusta el nombre de la última capa convolucional
    last_conv_layer_name = "conv2d_2"  # cambia si tu capa final tiene otro nombre
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    gradcam_img = display_gradcam(img, heatmap)

    st.image(gradcam_img, caption="Grad-CAM", use_column_width=True)

    # ---------------------------------------------------------
    # MAPA DE SALIENCIA (Saliency Map)
    # ---------------------------------------------------------
    st.subheader("⚡ Saliency Map")
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        top_class = tf.argmax(preds[0])
        top_class_channel = preds[:, top_class]

    grads = tape.gradient(top_class_channel, img_tensor)
    saliency = np.max(np.abs(grads[0]), axis=-1)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    fig, ax = plt.subplots()
    ax.imshow(np.array(img))
    ax.imshow(saliency, cmap="jet", alpha=0.4)
    ax.axis("off")
    st.pyplot(fig)

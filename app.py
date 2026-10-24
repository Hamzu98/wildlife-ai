
import os
import gradio as gr
import tensorflow as tf
import numpy as np

MODEL_PATH = "animal_detection_model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file animal_detection_model.keras nahi mili."
    )

model = tf.keras.models.load_model(MODEL_PATH)

# Ye order training ke waqt wali classes se match hona chahiye.
class_names = ["cats", "dogs", "elephents", "lions", "tigers"]


def predict_animal(image):
    if image is None:
        return {}, "Pehle animal ki photo upload karein."

    image = image.convert("RGB").resize((224, 224))
    image_array = np.array(image, dtype=np.float32)[None, ...]

    predictions = model.predict(image_array, verbose=0)[0]

    if len(predictions) != len(class_names):
        return {}, "Model ki classes aur class_names ki list match nahi kar rahi."

    results = {
        class_names[i].capitalize(): float(predictions[i])
        for i in range(len(class_names))
    }

    best_index = int(np.argmax(predictions))
    animal = class_names[best_index].capitalize()
    confidence = float(predictions[best_index]) * 100

    message = (
        f"### 🐾 Prediction: {animal}\n"
        f"**Confidence: {confidence:.1f}%**\n\n"
        "AI ki prediction ghalat bhi ho sakti hai."
    )

    return results, message


css = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
}
#hero {
    background: linear-gradient(135deg, #12372a, #176b50);
    color: white;
    text-align: center;
    padding: 28px 16px;
    border-radius: 20px;
}
#hero h1 {
    color: white !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="green"
    ),
    css=css,
    title="Wildlife AI"
) as demo:

    gr.HTML("""
    <div id="hero">
        <h1>🐾 Wildlife AI</h1>
        <p>Smart Animal Image Classification</p>
        <p>Upload a photo of a cat, dog, elephant, lion or tiger.</p>
    </div>
    """)

    gr.Markdown("## 🔍 Identify an Animal")

    with gr.Row():
        image = gr.Image(
            type="pil",
            label="Upload Animal Photo"
        )

        with gr.Column():
            result = gr.Label(
                num_top_classes=5,
                label="Prediction Results"
            )
            message = gr.Markdown(
                "Your prediction will appear here."
            )

    gr.Button(
        "✨ Identify Animal",
        variant="primary"
    ).click(
        fn=predict_animal,
        inputs=image,
        outputs=[result, message]
    )

    gr.Markdown(
        "---\n"
        "Wildlife AI · Powered by TensorFlow and Gradio"
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "10000"))
)

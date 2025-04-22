import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
import cv2
from PIL import Image

# Paths
model_path = os.path.join("model", "model.h5")
excel_file = "prediction_feedback.xlsx"
correction_dir = "prem's_corrections"
chart_path = "feedback_chart.png"

# Ensure directories exist
os.makedirs("model", exist_ok=True)
os.makedirs(correction_dir, exist_ok=True)

# Load model
model = load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Initialize Excel file


def init_excel():
    if not os.path.exists(excel_file):
        pd.DataFrame(columns=["ImageName", "Prediction", "Feedback",
                     "CorrectLabel"]).to_excel(excel_file, index=False)


init_excel()

# Predict function


def predict(image):
    image_np = np.array(image.convert("RGB"))
    resized = cv2.resize(image_np, (224, 224))
    img_array = img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "Cracked" if pred > 0.5 else "Not Cracked"
    return label

# Save feedback


def save_feedback(image, prediction, feedback, correct_label):
    image_name = getattr(image, 'name', 'UploadedImage')
    df = pd.read_excel(excel_file)

    # Save feedback
    new_row = {
        "ImageName": image_name,
        "Prediction": prediction,
        "Feedback": feedback,
        "CorrectLabel": correct_label if feedback == "Incorrect" else ""
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(excel_file, index=False)

    # Save incorrect image for retraining
    if feedback == "Incorrect":
        filename = f"{len(df)}_{correct_label}.png"
        save_path = os.path.join(correction_dir, filename)
        image.save(save_path)

    return generate_chart()

# Generate chart


def generate_chart():
    df = pd.read_excel(excel_file)
    counts = df["Feedback"].value_counts().reindex(
        ["Correct", "Incorrect"], fill_value=0)
    plt.figure(figsize=(4, 3))
    counts.plot(kind='bar', color=["green", "red"])
    plt.title("Prediction Feedback Summary")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return chart_path

# Reset feedback


def reset_feedback():
    pd.DataFrame(columns=["ImageName", "Prediction", "Feedback",
                 "CorrectLabel"]).to_excel(excel_file, index=False)
    return generate_chart()


# Gradio App
with gr.Blocks(title="Prem's Crack Detection") as demo:
    gr.Markdown(
        "<h2 style='text-align: center;'>Crack Detection with Feedback</h2>")

    # Row for uploading image
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")

    # Row for Predict button and showing prediction
    with gr.Row():
        predict_button = gr.Button("🔍 Predict")
        prediction_output = gr.Textbox(label="Prediction", interactive=False)

    # Row for feedback collection
    with gr.Row():
        feedback_radio = gr.Radio(
            ["Give Feedback", "Correct", "Incorrect"],
            label="Is the prediction correct?",
            value="Give Feedback"
        )

    with gr.Row():
        correct_label_dropdown = gr.Dropdown(
            choices=["Cracked", "Not Cracked"],
            label="Select the correct label (if Incorrect)",
            visible=False
        )

    # Row for submit and reset buttons
    with gr.Row():
        submit_button = gr.Button("✅ Submit Feedback")
        reset_button = gr.Button("🔁 Reset Feedback")

    # Graph output and download button
    graph_output = gr.Image(label="Feedback Graph")
    download_button = gr.File(
        value=excel_file, label="Download Feedback File")

    # Logic functions
    def handle_predict_button(image):
        if image is not None:
            return predict(image)
        return ""

    def toggle_dropdown(feedback_option):
        return gr.update(visible=(feedback_option == "Incorrect"))

    def handle_feedback_submit(feedback, image, prediction, correct_label):
        if feedback == "Give Feedback" or image is None:
            return gr.update(), excel_file
        if feedback == "Incorrect" and not correct_label:
            return gr.update(), excel_file
        chart = save_feedback(image, prediction, feedback, correct_label)
        return chart, excel_file

    # Events
    predict_button.click(fn=handle_predict_button,
                         inputs=image_input, outputs=prediction_output)
    feedback_radio.change(
        fn=toggle_dropdown, inputs=feedback_radio, outputs=correct_label_dropdown)
    submit_button.click(
        fn=handle_feedback_submit,
        inputs=[feedback_radio, image_input,
                prediction_output, correct_label_dropdown],
        outputs=[graph_output, download_button]
    )
    reset_button.click(fn=reset_feedback, outputs=graph_output)

# Uncomment this to run
# demo.launch()

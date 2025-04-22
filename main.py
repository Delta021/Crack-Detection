import gradio as gr
from prem_app import demo as prem_interface
from sasank_app import demo as sasank_interface

with gr.Blocks(title="Crack Detection App") as app:
    gr.Markdown("# 🧠 Crack Detection App")
    gr.Markdown(
        "Welcome! Explore crack detection using advanced machine learning models.")

    with gr.Tabs():
        with gr.TabItem("🏠 Home"):
            gr.Markdown("""
### 👋 Welcome to the Crack Detection App!

This app offers **two powerful models** to help you analyze and visualize cracks in images.

---

#### 🧠 Sasank's ResNet Model
- Based on **ResNet architecture** with custom Grad-CAM visualizations.
- Includes:
  - **Heatmaps** to highlight cracks
  - **Edge Detection**
  - **Bounding Boxes** on cracked regions
- Trained on a custom dataset for robust performance.

---

#### 👨‍🔬 Prem's Keras Model
- Lightweight and efficient **binary classifier**.
- Uses Keras and TensorFlow backend.
- Predicts whether an image is **Cracked** or **Not Cracked**.

Use the tabs above to try out each model!
""")

        with gr.TabItem("🧠 Sasank's Model"):
            sasank_interface.render()

        with gr.TabItem("👨‍🔬 Prem's Model"):
            prem_interface.render()
        '''with gr.TabItem("Tester"):
            tester_interface.render()'''

# Launch with better URL
app.launch(server_name="localhost", server_port=7862,
           share=False, show_error=True, inbrowser=True)

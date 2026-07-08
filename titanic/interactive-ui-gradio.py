from io import BytesIO
from pathlib import Path

import gradio as gr
import joblib
import hopsworks
import numpy as np
import requests
from PIL import Image

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model_reg = mr.get_model("titanic", version=1)
model_dir = model_reg.download()
model = joblib.load(model_dir + "/titanic_model.pkl")

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_BASE_URL = (
    "https://raw.githubusercontent.com/featurestorebook/mlfs-book/main/"
    "docs/titanic/assets/img"
)

# Hopsworks label_encoder uses alphabetical order: C=0, Q=1, S=2 and female=0, male=1.
# Gradio dropdown indices: Sex male=0/female=1, Embarked S=0/C=1/Q=2, Pclass First=0/Second=1/Third=2.
EMBARKED_ENCODING = {0: 2, 1: 0, 2: 1}
SEX_ENCODING = {0: 1, 1: 0}


def load_result_image(prediction: int) -> Image.Image:
    image_name = f"titanic_{int(prediction)}.jpg"
    local_path = REPO_ROOT / "docs/titanic/assets/img" / image_name
    if local_path.exists():
        return Image.open(local_path)

    response = requests.get(f"{IMAGE_BASE_URL}/{image_name}", timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def titanic(Sex, Age, Pclass, Fare, Parch, SibSp, Embarked):
    features = np.array([[
        Age,
        Pclass + 1.0,
        Fare,
        Parch,
        SibSp,
        EMBARKED_ENCODING[Embarked],
        SEX_ENCODING[Sex],
    ]])
    prediction = model.predict(features)[0]
    return load_result_image(prediction)


demo = gr.Interface(
    fn=titanic,
    title="Titanic Passenger Survival Predictive Analytics",
    description="Enter values for an imaginary passenger and the model will predict whether he/she survived or not.",
    inputs=[
        gr.Dropdown(choices=["male", "female"], type="index", label="Sex"),
        gr.Slider(minimum=1.0, maximum=100.0, step=1.0, label="Age"),
        gr.Dropdown(choices=["First", "Second", "Third"], type="index", label="Ticket class"),
        gr.Number(label="Fare ($)"),
        gr.Number(label="Number of parents/children aboard"),
        gr.Number(label="Number of siblings/spouses aboard"),
        gr.Dropdown(choices=["S", "C", "Q"], type="index", label="Port of Embarkation"),
    ],
    outputs=gr.Image(type="pil"),
    examples=[
        ["female", 30, "First", 22.10, 0.0, 0.0, "S"],
        ["male", 30, "Second", 8.11, 1.0, 1.0, "Q"],
    ],
)

demo.launch()

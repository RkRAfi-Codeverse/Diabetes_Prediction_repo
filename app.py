import gradio as gr
import pandas as pd
import pickle


with open("diabetes_best_model.pkl", "rb") as f:
    model = pickle.load(f)

#logic
def predict_diabetes(
    pregnancies, glucose, blood_pressure,
    skin_thickness, insulin, bmi,
    dpf, age
):
   
    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi,
        dpf, age
    ]], columns=[
        "Pregnancies", "Glucose", "BloodPressure", 
        "SkinThickness", "Insulin", "BMI", 
        "DiabetesPedigreeFunction", "Age"
    ])


    input_df['Glucose_BMI'] = input_df['Glucose'] * input_df['BMI']
    input_df['Age_Glucose'] = input_df['Age'] * input_df['Glucose']
    input_df['BloodPressure_BMI'] = input_df['BloodPressure'] * input_df['BMI']
    input_df['Insulin_Glucose_Ratio'] = input_df['Insulin'] / (input_df['Glucose'] + 1e-5)

  
    probability = model.predict_proba(input_df)[0][1]
    
    if probability >= 0.4:
        return f"###  Result: **Diabetic**\n"
    else:
        return f"###  Result: **Non-Diabetic**\n"


inputs = [
    gr.Slider(0, 12, step=1, label="Pregnancies"),
    gr.Slider(50, 350, step=1, label="Glucose Level (mg/dL)"),
    gr.Slider(40, 130, step=1, label="Blood Pressure (Low one)(mm Hg)"),
    gr.Slider(0, 100, step=1, label="Skin Thickness (mm)"),
    gr.Slider(0, 850, step=1, label="Insulin Level"),
    gr.Slider(10.0, 60.0, step=0.1, label="BMI"),
    gr.Slider(0.05, 2.5, step=0.01, label="Pedigree Func (Family History )(0-0.30 low, 0.50+ high)"),
    gr.Slider(10, 100, step=1, label="Age")
]


app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="markdown",
    title=" Diabetes Prediction System ",
    description=(
        "Enter medical parameters to check diabetes risk. "
        "This model uses a 0.4 threshold to prioritize patient safety."
    ),
    theme="soft"
)

if __name__ == "__main__":
    app.launch(share=True)
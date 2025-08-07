import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="ZyrexAn/DataScience-PromptEngineer-merged", device_map="auto")

def generate_code(prompt):
    return pipe(prompt, max_new_tokens=1000, do_sample=True)[0]['generated_text']

with gr.Blocks(theme=gr.themes.Base(), css=".footer {display:none;}") as demo:
    gr.Markdown(
        """
        # 🧑‍💻 Data Science Prompt Engineer

        💡 *Enter your data science question or coding prompt below and get instant Python code suggestions (powered by your custom AI model).*

        **Examples:**
        - *Load a CSV and plot a histogram of ages*
        - *Train a scikit-learn regression model and show RMSE*
        - *Clean missing values in a pandas DataFrame*
        """
    )
    with gr.Row():
        with gr.Column(scale=6):
            prompt = gr.Textbox(lines=2, label="Your Data Science Prompt")
        with gr.Column(scale=1):
            submit = gr.Button("Generate Code 💻")
    code_output = gr.Code(label="Generated Python Code", language="python", lines=15)
    gr.Examples(
        [
            ["Load a CSV and plot a boxplot of salaries"],
            ["Split a DataFrame into train/test and normalize all numeric columns"],
        ],
        prompt,
        fn=generate_code,
        outputs=code_output,
        cache_examples=False,
        label="Try these examples"
    )
    submit.click(generate_code, inputs=prompt, outputs=code_output)
    gr.Markdown(
        "✨ **Tip:** Copy the generated code into your Python script or notebook and adapt as needed."
    )

demo.launch()

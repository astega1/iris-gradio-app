import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Download your GGUF model from HF Hub
model_path = hf_hub_download(
    repo_id="astegaras/lora_python_converter",
    filename="llama-3.2-3b-instruct.Q2_K.gguf"
)

# Load GGUF with safe HF settings
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=4,
    n_batch=64,
    n_gpu_layers=0,     # IMPORTANT
    use_mmap=False,     # IMPORTANT
    use_mlock=False,    # IMPORTANT
    low_vram=True,      # IMPORTANT
    verbose=False
)

def generate_code(instruction):
    messages = [
        {"role": "system", "content": "You are a Python code generator. Return only code."},
        {"role": "user", "content": instruction},
    ]

    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.2,
        top_p=0.5
    )

    return out["choices"][0]["message"]["content"]

# ---- GRADIO UI ----
with gr.Blocks(theme="gradio/soft") as demo:
    gr.Markdown(
        """
        # Python Code Generator  
        Enter a task in plain English and receive executable Python code.

        Example:  
        *"Help me set up my to-do list"*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            instruction = gr.Textbox(
                label="Describe what you want to build",
                placeholder="Example: Help me set up my to-do list",
                lines=3,
            )
            submit = gr.Button("Generate Python Code", variant="primary")

        with gr.Column(scale=1):
            code_output = gr.Code(
                label="Generated Python Code",
                language="python"
            )

    submit.click(fn=generate_code, inputs=instruction, outputs=code_output)

demo.launch(share=True)



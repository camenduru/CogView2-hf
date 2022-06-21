import os
import gradio as gr
os.system("pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html")
os.system("git clone https://github.com/Sleepychord/Image-Local-Attention")
os.chdir("Image-Local-Attention")
os.system("python setup.py install")
os.chdir("..")
os.system("git clone https://github.com/NVIDIA/apex")
os.chdir("apex")
os.system('pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./')
os.chdir("..")
os.system("git clone https://github.com/THUDM/CogView2")
os.chdir("CogView2")
os.system("gdown https://drive.google.com/uc?id=1-2nI2TTUOdiQ2WpydGafk_bZIZggQBK4")
os.system("7za x coglm.zip")
os.system("gdown https://drive.google.com/uc?id=1ulfXJFstYZUestvWcQIadKkNNDVbpIdM")

def inference(text):
    with open("input.txt") as f:
        lines = f.readlines()
    lines[0] = text
    with open("input.txt", "w") as f:
        f.writelines(lines)
    os.system("python cogview2_text2image.py --mode inference --fp16 --input-source input.txt --output-path samples_sat_v0.2 --batch-size 4 --max-inference-batch-size 8 --only-first-stage")
    return "/content/CogView2/output.png"

gr.Interface(inference,"text","image",title="CogView 2").launch(debug=True,enable_queue=True)






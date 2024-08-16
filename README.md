in progress...
Notes:
For Inference Server: 2070 with Cuda 12.1, did instructions here: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
.env file needs HUGGING_FACE_API_TOKEN for gated repos (ex Llama3.1)
.env file needs MODELS - comma separated list of model names
.env file needs BACKEND_URL
.env file can have HF_CACHE_DIR


streamlit run app.py
flask --app backend.py run
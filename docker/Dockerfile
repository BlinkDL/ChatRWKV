FROM nvcr.io/nvidia/pytorch:23.02-py3
LABEL org.opencontainers.image.authors="soulteary@gmail.com"

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install huggingface_hub
WORKDIR /app
RUN cat > /get-models.py <<EOF
from huggingface_hub import hf_hub_download
title = "RWKV-4-Pile-14B-20230313-ctx8192-test1050"
model_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-14b", filename=f"{title}.pth")
EOF

RUN python /get-models.py && \
    rm -rf /get-models.py && \
    pip install ninja gradio && \
    pip install rwkv==0.7.1 --index-url=https://pypi.org/simple

RUN git clone https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio/ ChatRWKV && \
    cd ChatRWKV && git checkout 699bb36437cd15ef8da05c4f392f4ab3ea6be65d
WORKDIR /app/ChatRWKV

COPY docker/webui.py /app/ChatRWKV/app.py
COPY docker/*.mini.py /app/ChatRWKV/
CMD ["python", "/app/ChatRWKV/app.py"]
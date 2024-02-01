FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV LANG C.UTF-8

RUN echo "Asia/Shanghai" > /etc/timezone

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime


RUN apt-get update  && apt-get install ffmpeg vim curl git -y

WORKDIR /app

COPY ./resources ./resources

COPY requirements.txt .

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./whisperx ./whisperx
COPY ./app ./app
COPY ./setup.py .
COPY ./wsgi.py .

RUN pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD ["python3", "wsgi.py"]

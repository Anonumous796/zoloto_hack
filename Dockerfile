FROM python:3.7

RUN rm -rf /var/lib/apt/lists/* && apt-get clean && apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

RUN pip install fastapi uvicorn torch sahi pycocotools pillow==6.2.2 colorama torchvision pandas albumentations python-multipart

EXPOSE 5000

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
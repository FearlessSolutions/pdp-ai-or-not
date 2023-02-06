FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN pip install pandas numpy scipy

ENTRYPOINT jupyter notebook --notebook-dir=/home/jovyan/work --ip 0.0.0.0 --no-browser --allow-root --port=8888

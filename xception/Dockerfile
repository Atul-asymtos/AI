# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Install required packages
#RUN apt-get update && \
#    apt-get install -y libgl1-mesa-glx && \
#    pip install --no-cache-dir flask opencv-python-headless tensorflow
# Install OpenCV
RUN apt-get update
#RUN apt-get install -y libopencv-dev libgtk2.0-dev pkg-config opencv-python-headless
RUN apt-get update && apt-get install -y build-essential cmake libgtk2.0-dev pkg-config
# Download and build OpenCV from source
RUN git clone https://github.com/opencv/opencv.git && \
    cd opencv && mkdir build && cd build && \
    cmake .. && make -j4 && make install

# Clean up by removing the source code and build files
RUN rm -rf /app/opencv

# Install TensorRT
#RUN apt-get update && apt-get install -y nvidia-cuda-toolkit
# nvidia-tensorrt

# Set the TF_TRT_LIBRARY environment variable
#ENV TF_TRT_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer.so
# Expose the port the application runs on
EXPOSE 5000

# Define the command to run your application
CMD ["python", "xception.py"]

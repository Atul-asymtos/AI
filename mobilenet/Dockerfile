# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip install --upgrade Flask Werkzeug
RUN pip install pillow

# Copy the current directory contents into the container at /app
COPY . /app

# Expose port 5000 for the Flask application
EXPOSE 5000

# Define environment variables
ENV FLASK_APP=mobilenet.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask application
CMD ["flask", "run"]

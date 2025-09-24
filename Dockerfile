# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY bert_training/requirements.txt /app/

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to reduce layer size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Default command to execute when the container starts
# This can be overridden in the `docker run` command
# For example, to run training:
# docker run <image_name> python bert_training/train.py --mode unsupervised ...
CMD ["/bin/bash"]

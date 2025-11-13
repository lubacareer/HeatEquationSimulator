# Use the TensorFlow official GPU image as the base image
FROM tensorflow/tensorflow:2.15.0-gpu

# Set the working directory
WORKDIR /app

# Copy the executable to the container
COPY dist/interface3.exe /app/interface3.exe

# Copy the Python source files to the container
COPY interface3.py /app/interface3.py
COPY deepLearning.py /app/deepLearning.py
COPY backend.py /app/backend.py
COPY requirements.txt /app/requirements.txt
COPY README.md /app/README.md

# Install Tkinter and other dependencies
RUN apt-get update && apt-get install -y python3-tk

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to run your application
CMD ["python3", "interface3.py"]

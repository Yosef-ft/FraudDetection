# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /flask_app/app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt 

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run serve_model.py when the container launches
CMD ["python", "-m", "flask","--app", "routes", "run", "--host=0.0.0.0", "--port=5000"]
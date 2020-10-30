FROM python:3.7.3-stretch

# Make the container's port 8080 available to the outside world
EXPOSE 8080

# Working Directory
WORKDIR /app

# Copy source code to working directory
ADD . /app/

# Install packages from requirements.txt
# hadolint ignore=DL3013
RUN pip install --upgrade pip &&\
    pip install --trusted-host pypi.python.org -r requirements.txt

# RUN
CMD ["python", "main.py"]


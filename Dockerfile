# Base image is debian:bullseye-slim (Debian 11), but bookworm is (Debian 12)
FROM debian:bullseye-slim

# Step 0: Install python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && apt-get clean

# Step 1: Set the working directory
WORKDIR /code

# Step 2: Copy the requirements.txt file to the working directory
COPY ./requirements.txt /code/requirements.txt

# Step 3: Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Step 4: Copy the app directory to the working directory
COPY ./app /code/app

# Step 5: Run the container as a non-root user
USER nobody

# Step 6: Run uvicorn with the app
CMD ["uvicorn", "code.main:app", "--host", "0.0.0.0", "--port", "8000"]
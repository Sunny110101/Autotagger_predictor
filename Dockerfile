FROM python:3.9-slim-bullseye

# Set environment variables
ENV APP=/app

WORKDIR $APP

# Install system dependencies
RUN apt-get update && apt-get install -y gcc python3-dev

# Create requirements.txt with necessary packages
RUN echo "sentence-transformers\n\
numpy\n\
tqdm\n\
torch" > requirements.txt

# Install Python packages including sentence-transformers
RUN pip install -r requirements.txt --no-cache-dir

# Create models directory
RUN mkdir -p $APP/models/sentence-transformers_all-MiniLM-L6-v2

# Now download and save the model after sentence-transformers is installed
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('all-MiniLM-L6-v2'); \
    model.save('$APP/models/sentence-transformers_all-MiniLM-L6-v2')"

# Copy source code
COPY src/* $APP/

# Make sure the script is executable
RUN chmod +x tagtest2.py

# Set the entrypoint
ENTRYPOINT ["python"]
CMD ["tagtest2.py"]
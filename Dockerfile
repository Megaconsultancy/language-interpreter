# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file first (to leverage Docker cache)
COPY requirements.txt .

# Install dependencies (using the CPU-only version of PyTorch to keep the image size small)
RUN pip install --no-cache-dir -r requirements.txt

# To speed up the first startup in the container, we can pre-download the model during the Docker build
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; model_name='facebook/nllb-200-distilled-600M'; AutoTokenizer.from_pretrained(model_name, src_lang='eng_Latn'); AutoModelForSeq2SeqLM.from_pretrained(model_name)"

# Copy the API code
COPY api.py .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

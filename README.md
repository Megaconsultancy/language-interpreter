# Bemba & Chewa Translation API

This is a fast, local translation API built with FastAPI that uses the `facebook/nllb-200-distilled-600M` AI model to translate English text into Bemba or Chewa (Nyanja). 

The API loads the AI model into memory exactly once at startup, allowing for blazing fast translations (< 300ms) without the overhead of loading a 2.4GB model on every request.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Running the API

The easiest way to run the API is using Docker Compose. The `Dockerfile` is configured to pre-download the AI model during the build phase, so the image will be fully self-contained.

1. **Build and start the container in the background:**
   ```bash
   docker compose up -d --build
   ```
   *(Note: The first time you run this, it will take several minutes to download PyTorch and the 2.4GB translation model into the Docker image).*

2. **Check the logs to see when the API is ready:**
   ```bash
   docker compose logs -f
   ```
   Wait until you see the message: `✅ Model loaded! API is ready to accept requests.` You can press `Ctrl+C` to exit the logs.

3. **To stop the API when you are done:**
   ```bash
   docker compose down
   ```

## Testing the API

Once the API is running on port 8000, you can test it by sending a `POST` request to the `/translate` endpoint. 

You must provide a JSON body with the `text` you want to translate, and the target `lang` (`bem` for Bemba, or `nya` for Chewa/Nyanja).

### Example: Translating to Bemba

```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Where is the closest hospital?", "lang": "bem"}'
```

**Response:**
```json
{
  "original": "Where is the closest hospital?",
  "translation": "Bushe cipatala icapalamishe caba kwi?",
  "lang": "bem"
}
```

### Example: Translating to Chewa (Nyanja)

```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Where is the closest hospital?", "lang": "nya"}'
```

**Response:**
```json
{
  "original": "Where is the closest hospital?",
  "translation": "Kodi chipatala chapafupi chili kuti?",
  "lang": "nya"
}
```

## Running Locally (Without Docker)

If you prefer to run the API directly on your machine without Docker:

1. Create a virtual environment and install the dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Start the FastAPI server:
   ```bash
   python api.py
   ```

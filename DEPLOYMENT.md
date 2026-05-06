# Deployment

This project is a Streamlit web app, but it is configured for Render Docker deployment so it does not depend on Streamlit Community Cloud.

## Render Deployment

1. Open https://dashboard.render.com.
2. Choose **New** -> **Blueprint**.
3. Connect `Rudwpahs/shooting-form-analysis`.
4. Render will detect `render.yaml` and build the app from `Dockerfile`.
5. After deployment, open the generated `*.onrender.com` URL.

The Render service runs:

```bash
streamlit run web_app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

This matters because Render web services must listen on `0.0.0.0` and use Render's `PORT` environment variable.

## Files Used For Deployment

- `Dockerfile`: builds a Python 3.12 image and starts the Streamlit app.
- `render.yaml`: creates a Render Docker web service with auto-deploy on commit.
- `requirements.txt`: Python dependencies.
- `packages.txt`: Linux packages needed by OpenCV and MediaPipe.

## Local Docker Test

```bash
docker build -t shooting-form-analysis .
docker run --rm -p 10000:10000 -e PORT=10000 shooting-form-analysis
```

Then open:

```text
http://localhost:10000
```

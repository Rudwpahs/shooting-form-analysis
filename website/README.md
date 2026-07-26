# Shooting Form Studio

This folder contains the browser-based companion site for the shooting-form
analysis project. It runs pose estimation locally in the browser with MediaPipe,
so uploaded clips are not sent to a server.

## Run locally

```bash
npm install
npm run dev
```

Open the local URL printed by the dev server, upload a basketball clip, and use
the timeline to inspect release detection and joint-angle metrics. The app ships
with a MediaPipe pose model under `public/models/` and has no API key requirement.

The original Streamlit workflow at the repository root remains available for
batch and notebook-oriented analysis.

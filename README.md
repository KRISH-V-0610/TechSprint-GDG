<div align="center">

  # ShushrutAI v2
  <h3>Advanced Dermatological Intelligence</h3>

  <p>
    <img src="https://img.shields.io/badge/Built_With-React_|_Node_|_Python-purple?style=for-the-badge" alt="Stack">
    <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
  </p>

  <p>
    <b>Revolutionizing Dermatology with Multi-Agent AI</b><br>
    <i>Detect. Diagnose. Treat. Voice-Assisted.</i>
  </p>
</div>

---

## About The Project

**ShushrutAI** is a state-of-the-art dermatological platform designed to empower doctors with instant, AI-driven skin analysis. Named after *Sushruta*, the father of plastic surgery, this tool blends ancient wisdom with cutting-edge Generative AI.

It employs a **Multi-Agent System** where specialized AI agents collaborate to verify symptoms, cross-reference diseases, generate comprehensive medical reports, and provide voice-guided consultation via "Jarvis".

### Key Features

- **Precision Diagnostics:** Hybrid analysis using DenseNet121 and Google Gemini
- **Multi-Agent Architecture:**
  - **Verify Agent:** Filters non-skin images and assesses general health
  - **Pathology Agent:** Detailed analysis of unhealthy skin conditions
  - **Report Agent:** Generates professional-grade medical reports
  - **Jarvis Agent:** Voice-interactive assistant for doctors to discuss cases
- **PDF Report Generation:** Automatic, downloadable PDF reports for patient records
- **Patient Management:** Register patients, manage records, upload and track skin images

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite, Tailwind CSS, Zustand, Firebase Auth |
| Backend | Node.js, Express, Firebase Admin, Cloudinary |
| ML Service | FastAPI, PyTorch (DenseNet121), Google Gemini |
| Database | Firestore |
| Storage | Cloudinary |

---

## The Multi-Agent Workflow

1. **Input:** Doctor uploads a patient's skin image
2. **Models:** DenseNet121 provides initial classification (23 disease classes)
3. **Agent 1 (Verify):** Confirms image validity and assesses general skin health
4. **Agent 2 (Diagnosis):** Deep dives into potential pathologies if unhealthy
5. **Agent 3 (Report):** Synthesizes all data into a structured clinical report
6. **Agent 4 (Jarvis):** Reads the report and stands by for Q&A

---

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.10+ (Conda recommended)
- Firebase project with Firestore enabled
- Cloudinary account
- Google Gemini API key

### 1. Clone the repository

```sh
git clone https://github.com/Asc-85129/shushrutAI.git
cd shushrutAI
```

### 2. Install all JS dependencies

```sh
npm install
# Automatically installs dependencies for frontend and backend too
```

### 3. Install Python dependencies

```sh
cd PYTHON
pip install -r requirements.txt
```

### 4. Set up environment variables

**`frontend/.env`** — Firebase and Cloudinary credentials (see `frontend/.env.example`):
```
VITE_FIREBASE_API_KEY=
VITE_FIREBASE_AUTH_DOMAIN=
VITE_FIREBASE_PROJECT_ID=
VITE_FIREBASE_STORAGE_BUCKET=
VITE_FIREBASE_MESSAGING_SENDER_ID=
VITE_FIREBASE_APP_ID=
VITE_CLOUDINARY_CLOUD_NAME=
VITE_CLOUDINARY_UPLOAD_PRESET=
```
> API URLs are set automatically — localhost in dev, deployed URLs in production build.

**`backend/.env`** — (see `backend/.env.example`):
```
PORT=5000
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=
CLOUDINARY_UPLOAD_PRESET=
PYTHON_API_URL=http://127.0.0.1:6700
```
Place your Firebase service account key at `backend/serviceAccountKey.json`.

**`PYTHON/.env`**:
```
GOOGLE_API_KEY=
PORT=6700
```

### 5. Start all services

```sh
npm run dev
```

This starts all three services at once:
- Frontend → http://localhost:5173
- Backend → http://localhost:5000
- Python ML → http://localhost:6700 (API docs: http://localhost:6700/docs)

---

## Deployment

Backend and Python services are deployed on [Render](https://render.com). Set production environment variables via the Render dashboard. Frontend is deployed via Vercel/Netlify — set Firebase and Cloudinary vars in the platform dashboard.

---

## Contributors

**Team DOMinators**

- Nisarg
- Vansh
- Krish
- Heli
- Rahul

---

<div align="center">
  <small>Made with ❤️ by Team DOMinators</small>
</div>

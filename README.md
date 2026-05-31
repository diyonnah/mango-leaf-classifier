# Mango Leaf Classifier (Next.js + Vercel)

## Overview
Next.js frontend with a Python `/api/predict` endpoint on Vercel. The model file is loaded from the repo root.

## Local Development
1. Install deps:
   ```bash
   npm install
   ```
2. Run the Python API (local):
   ```bash
   python app.py
   ```
3. Point the UI to the local API:
   ```bash
   export NEXT_PUBLIC_PREDICT_URL=http://127.0.0.1:5000/predict
   ```
   Windows PowerShell:
   ```powershell
   $env:NEXT_PUBLIC_PREDICT_URL="http://127.0.0.1:5000/predict"
   ```
4. Run Next.js:
   ```bash
   npm run dev
   ```

## Deploy to Vercel (GitHub)
1. Push this repo to GitHub.
2. Go to https://vercel.com/new and import the repo.
3. Framework: Next.js (auto-detect).
4. Deploy.

## Notes
- Ensure `MANGO_LEAF_Classifier.sav` stays in the repo root.
- The API endpoint is at `/api/predict` after deployment.

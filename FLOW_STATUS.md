# Mango Leaf Classifier - Flow and Status

## Flow
1. User uploads a mango leaf image in the Next.js UI.
2. UI previews the image and sends a POST request with the file.
3. `/api/predict` (Python on Vercel) loads the model and extracts features.
4. Model returns `Healthy` or `Unhealthy` with optional confidence.
5. UI displays result, emoji, and confidence bar.

## Current Status
- [x] Next.js app structure created (app router with custom styling).
- [x] UI ported from the original index.html with matching layout and behavior.
- [x] Prediction API implemented in `/api/predict.py` using the existing model file.
- [x] Vercel runtime config added for the Python API.
- [ ] Local dev flow documented and verified.
- [ ] Vercel deployment created and tested.

## Next Updates
- [ ] Install Node deps: `npm install`.
- [ ] For local dev, run `python app.py` and set `NEXT_PUBLIC_PREDICT_URL=http://127.0.0.1:5000/predict`.
- [ ] Run `npm run dev` and test upload + prediction.
- [ ] Deploy to Vercel and confirm `/api/predict` works in production.

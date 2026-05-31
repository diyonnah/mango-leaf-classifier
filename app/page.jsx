"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = "/api/predict";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewName, setPreviewName] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const resultRef = useRef(null);

  useEffect(() => {
    if (!resultRef.current || !result) {
      return;
    }
    resultRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [result]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) {
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const url = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(url);
    setPreviewName(file.name);
    setResult(null);
  };

  const handleClassify = async () => {
    if (!selectedFile || loading) {
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch(API_URL, { method: "POST", body: formData });
      const contentType = response.headers.get("content-type") || "";
      const data = contentType.includes("application/json")
        ? await response.json()
        : { error: await response.text() };

      if (!response.ok && !data.error) {
        data.error = `Request failed with status ${response.status}`;
      }

      if (data.error) {
        setResult({
          type: "error",
          emoji: "⚠️",
          label: "Error",
          desc: data.error,
          confidence: null
        });
      } else {
        const lowerResult = String(data.result || "").toLowerCase();
        const isHealthy = lowerResult === "healthy";
        setResult({
          type: isHealthy ? "alive" : "dead",
          emoji: isHealthy ? "🌿" : "🍂",
          label: data.result || "Unknown",
          desc: isHealthy
            ? "The leaf appears healthy."
            : "The leaf appears unhealthy or diseased.",
          confidence: data.confidence
        });
      }
    } catch (error) {
      setResult({
        type: "error",
        emoji: "⚠️",
        label: "Connection Error",
        desc: `Could not reach the server. ${error?.message || "Is the API running?"}`,
        confidence: null
      });
    } finally {
      setLoading(false);
    }
  };

  const confidence = result?.confidence;
  const showConfidence = confidence !== null && confidence !== undefined;

  return (
    <div className="container">
      <header>
        <span className="leaf-icon">🍃</span>
        <h1>
          Mango Leaf <span>Classifier</span>
        </h1>
        <p className="subtitle">SVM-powered leaf health detection</p>
      </header>

      <div className="card">
        <div
          className={`upload-zone${dragOver ? " dragover" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragOver(false);
            const file = event.dataTransfer.files?.[0];
            if (file) {
              handleFile(file);
            }
          }}
        >
          <input
            type="file"
            accept="image/*"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                handleFile(file);
              }
            }}
          />
          <span className="upload-icon">📁</span>
          <p className="upload-title">Drop a leaf image here</p>
          <p className="upload-hint">or click to browse — JPG, PNG supported</p>
        </div>

        <div
          className="preview-wrap"
          style={{ display: previewUrl ? "block" : "none" }}
        >
          <img src={previewUrl} alt="Preview" />
          <p className="preview-name">{previewName}</p>
        </div>

        <button
          className={`btn${loading ? " loading" : ""}`}
          onClick={handleClassify}
          disabled={!selectedFile || loading}
        >
          {loading ? (
            <>
              <span className="spinner" /> Analyzing...
            </>
          ) : (
            "Classify Leaf"
          )}
        </button>

        <div
          ref={resultRef}
          className={`result-wrap${result ? " " + result.type : ""}`}
          style={{ display: result ? "block" : "none" }}
        >
          <span className="result-emoji">{result?.emoji}</span>
          <div className="result-label">{result?.label}</div>
          <div className="result-desc">{result?.desc}</div>
          <div
            className="confidence-bar-wrap"
            style={{ display: showConfidence ? "block" : "none" }}
          >
            <div className="confidence-label">Confidence: {confidence}%</div>
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: showConfidence ? `${confidence}%` : "0%" }}
              />
            </div>
          </div>
        </div>

        <hr className="divider" />

        <div className="how-title">How it works</div>
        <div className="steps">
          <div className="step">
            <div className="step-num">01</div>
            <div className="step-text">Upload a photo of a mango leaf</div>
          </div>
          <div className="step">
            <div className="step-num">02</div>
            <div className="step-text">SVM model analyzes the image features</div>
          </div>
          <div className="step">
            <div className="step-num">03</div>
            <div className="step-text">
              Get instant Healthy or Unhealthy prediction
            </div>
          </div>
        </div>
      </div>

      <footer>Mango Leaf Health Classifier · Powered by SVM + Flask</footer>
    </div>
  );
}

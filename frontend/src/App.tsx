import React, { useState } from "react";

interface AlprResult {
  plate_text: string;
  confidence?: number;
  plate_image?: string;
  annotated_image?: string;
}

const API_BASE = "http://localhost:8000";

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AlprResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setSelectedFile(file);
    setResult(null);
    setError(null);

    if (file) {
      setPreview(URL.createObjectURL(file));
    } else {
      setPreview(null);
    }
  };

  const uploadImage = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const form = new FormData();
      form.append("image", selectedFile);

      const res = await fetch(`${API_BASE}/detect`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        throw new Error(`HTTP error ${res.status}`);
      }

      const json = await res.json();

      if (json.success && json.data?.success) {
        setResult(json.data as AlprResult);
      } else if (json.success && !json.data?.success) {
        setError(json.data.error || "Backend error");
      } else {
        setError("Unexpected response from backend");
      }
    } catch (err) {
      console.error(err);
      setError("Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  };

  const plateImgUrl =
    result?.plate_image != null
      ? `${API_BASE}/media/${result.plate_image}`
      : null;
  const annotatedImgUrl =
    result?.annotated_image != null
      ? `${API_BASE}/media/${result.annotated_image}`
      : null;

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#f5f7fb",
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        paddingTop: "40px",
        fontFamily: "Segoe UI, system-ui, sans-serif",
      }}
    >
      <div
        style={{
          width: "90%",
          maxWidth: "1100px",
          backgroundColor: "#ffffff",
          padding: "30px 40px 40px",
          borderRadius: "16px",
          boxShadow: "0 14px 45px rgba(0,0,0,0.08)",
        }}
      >
        <h1
          style={{
            textAlign: "center",
            marginBottom: "10px",
            fontSize: "32px",
            letterSpacing: "1px",
          }}
        >
          Automatic License Plate Recognition
        </h1>
        <p
          style={{
            textAlign: "center",
            color: "#666",
            marginBottom: "30px",
          }}
        >
          Upload a car image and the system will detect the plate, crop it, and
          read the characters automatically.
        </p>

        {/* Top area: upload + original preview */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1.2fr 1.8fr",
            gap: "30px",
            alignItems: "center",
          }}
        >
          <div>
            <label
              style={{
                display: "inline-block",
                padding: "10px 18px",
                borderRadius: "999px",
                backgroundColor: "#eee",
                cursor: "pointer",
                marginBottom: "12px",
                fontSize: "14px",
              }}
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />
              Choose File
            </label>
            {selectedFile && (
              <span style={{ marginLeft: "10px", color: "#444" }}>
                {selectedFile.name}
              </span>
            )}

            <div style={{ marginTop: "20px" }}>
              <button
                onClick={uploadImage}
                disabled={!selectedFile || loading}
                style={{
                  padding: "10px 26px",
                  backgroundColor: "#007bff",
                  color: "white",
                  border: "none",
                  borderRadius: "999px",
                  cursor: !selectedFile || loading ? "not-allowed" : "pointer",
                  opacity: !selectedFile || loading ? 0.6 : 1,
                  fontSize: "15px",
                  fontWeight: 500,
                }}
              >
                {loading ? "Processing..." : "Upload & Detect"}
              </button>
            </div>

            {error && (
              <div
                style={{
                  marginTop: "18px",
                  padding: "10px 14px",
                  borderRadius: "8px",
                  backgroundColor: "#ffe5e5",
                  color: "#b00020",
                  fontSize: "14px",
                }}
              >
                {error}
              </div>
            )}

            {result && !error && (
              <div
                style={{
                  marginTop: "22px",
                  padding: "14px 18px",
                  borderRadius: "10px",
                  backgroundColor: "#e6fff0",
                  border: "1px solid #b2f0ce",
                }}
              >
                <div
                  style={{
                    fontSize: "20px",
                    fontWeight: 600,
                    color: "#1b8c3a",
                    marginBottom: "6px",
                  }}
                >
                  Plate Detected:{" "}
                  <span style={{ fontFamily: "monospace" }}>
                    {JSON.stringify(result.plate_text)}
                  </span>
                </div>
                {result.confidence != null && (
                  <div style={{ fontSize: "13px", color: "#3b7f52" }}>
                    Detection confidence:{" "}
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                )}
                <div
                  style={{
                    fontSize: "12px",
                    color: "#777",
                    marginTop: "6px",
                  }}
                >
                  The system first detects the plate location using YOLO, then
                  crops the region and applies OCR to read the characters.
                </div>
              </div>
            )}
          </div>

          <div
            style={{
              justifySelf: "center",
            }}
          >
            {preview ? (
              <img
                src={preview}
                alt="preview"
                style={{
                  maxWidth: "100%",
                  maxHeight: "420px",
                  borderRadius: "12px",
                  boxShadow: "0 10px 30px rgba(0,0,0,0.12)",
                }}
              />
            ) : (
              <div
                style={{
                  width: "100%",
                  height: "280px",
                  borderRadius: "12px",
                  background:
                    "repeating-linear-gradient(45deg,#f1f1f1,#f1f1f1 10px,#fafafa 10px,#fafafa 20px)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#999",
                  fontSize: "14px",
                }}
              >
                No image selected yet.
              </div>
            )}
          </div>
        </div>

        {/* Output images */}
        {result && !error && (plateImgUrl || annotatedImgUrl) && (
          <div
            style={{
              marginTop: "35px",
              borderTop: "1px solid #eee",
              paddingTop: "24px",
            }}
          >
            <h2
              style={{
                fontSize: "20px",
                marginBottom: "16px",
                color: "#333",
              }}
            >
              Detection Outputs
            </h2>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                gap: "20px",
              }}
            >
              {annotatedImgUrl && (
                <div>
                  <div
                    style={{
                      marginBottom: "6px",
                      fontSize: "14px",
                      fontWeight: 500,
                    }}
                  >
                    Car with Detected Plate (YOLO)
                  </div>
                  <img
                    src={annotatedImgUrl}
                    alt="annotated car"
                    style={{
                      width: "100%",
                      borderRadius: "10px",
                      boxShadow: "0 10px 30px rgba(0,0,0,0.08)",
                    }}
                  />
                </div>
              )}

              {plateImgUrl && (
                <div>
                  <div
                    style={{
                      marginBottom: "6px",
                      fontSize: "14px",
                      fontWeight: 500,
                    }}
                  >
                    Cropped Plate Used for OCR
                  </div>
                  <img
                    src={plateImgUrl}
                    alt="plate crop"
                    style={{
                      width: "100%",
                      borderRadius: "10px",
                      boxShadow: "0 10px 30px rgba(0,0,0,0.08)",
                    }}
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

export const runtime = "nodejs";

function buildBackendCandidates(rawUrl) {
  const base = new URL(rawUrl);
  const paths = [base.pathname || "/", "/predict", "/api/predict"];
  const uniquePaths = [...new Set(paths)];

  return uniquePaths.map((pathname) => {
    const candidate = new URL(base.toString());
    candidate.pathname = pathname;
    return candidate.toString();
  });
}

async function forwardPrediction(request) {
  const backendUrl =
    process.env.PREDICT_API_URL || process.env.NEXT_PUBLIC_PREDICT_URL;

  if (!backendUrl) {
    return Response.json(
      {
        error:
          "Prediction backend is not configured. Set PREDICT_API_URL to your deployed inference service."
      },
      { status: 501 }
    );
  }

  const formData = await request.formData();
  const image = formData.get("image") || formData.get("file");

  if (!image) {
    return Response.json(
      { error: "No image uploaded. Use form field 'image' or 'file'." },
      { status: 400 }
    );
  }

  const proxyFormData = new FormData();
  proxyFormData.append("image", image);

  let lastPayload = null;
  let lastStatus = 502;

  for (const candidateUrl of buildBackendCandidates(backendUrl)) {
    const response = await fetch(candidateUrl, {
      method: "POST",
      body: proxyFormData
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : { error: await response.text() };

    lastPayload = payload;
    lastStatus = response.status;

    if (response.ok && !payload?.error) {
      return Response.json(payload, { status: response.status });
    }

    if (response.status !== 404) {
      return Response.json(payload, { status: response.status });
    }
  }

  return Response.json(
    {
      error:
        lastPayload?.error ||
        "Prediction backend returned 404 on all tried endpoints.",
      tried: buildBackendCandidates(backendUrl)
    },
    { status: lastStatus }
  );
}

export async function POST(request) {
  try {
    return await forwardPrediction(request);
  } catch (error) {
    return Response.json(
      {
        error: `Prediction request failed: ${error?.message || "Unknown error"}`
      },
      { status: 500 }
    );
  }
}
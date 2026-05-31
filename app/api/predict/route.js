export const runtime = "nodejs";

function resolveBackendEndpoint(rawUrl) {
  const endpoint = new URL(rawUrl);

  if (!endpoint.pathname || endpoint.pathname === "/") {
    endpoint.pathname = "/predict";
  }

  return endpoint.toString();
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

  const response = await fetch(resolveBackendEndpoint(backendUrl), {
    method: "POST",
    body: proxyFormData
  });

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : { error: await response.text() };

  return Response.json(payload, { status: response.status });
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
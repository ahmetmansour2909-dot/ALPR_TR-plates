import { serve } from "https://deno.land/std@0.203.0/http/server.ts";
import {
  dirname,
  fromFileUrl,
  join,
} from "https://deno.land/std@0.203.0/path/mod.ts";

const PORT = 8000;
const PYTHON_CMD = "py"; // كما قلت، تستخدم py في جهازك

const BACKEND_DIR = dirname(fromFileUrl(import.meta.url));
const UPLOAD_DIR = join(BACKEND_DIR, "uploads");
const PYTHON_SCRIPT = join(dirname(BACKEND_DIR), "python", "alpr_yolo.py");

await Deno.mkdir(UPLOAD_DIR, { recursive: true });

const CORS_HEADERS: HeadersInit = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
};

const JSON_HEADERS: HeadersInit = {
  ...CORS_HEADERS,
  "Content-Type": "application/json",
};

async function handleDetect(req: Request): Promise<Response> {
  const contentType = req.headers.get("content-type") ?? "";
  if (!contentType.includes("multipart/form-data")) {
    return new Response(
      JSON.stringify({ success: false, error: "Expected multipart/form-data" }),
      { status: 400, headers: JSON_HEADERS },
    );
  }

  const form = await req.formData();
  const file = form.get("image");

  if (!(file instanceof File)) {
    return new Response(
      JSON.stringify({ success: false, error: "Field 'image' is required" }),
      { status: 400, headers: JSON_HEADERS },
    );
  }

  const bytes = new Uint8Array(await file.arrayBuffer());
  const safeName = file.name && file.name !== "blob" ? file.name : "upload.jpg";
  const filename = `${Date.now()}_${safeName}`;
  const filePath = join(UPLOAD_DIR, filename);

  await Deno.writeFile(filePath, bytes);

  const cmd = new Deno.Command(PYTHON_CMD, {
    args: [PYTHON_SCRIPT, filePath],
    stdout: "piped",
    stderr: "piped",
  });

  const { code, stdout, stderr } = await cmd.output();

  if (code !== 0) {
    const errText = new TextDecoder().decode(stderr);
    console.error("Python error:", errText);
    return new Response(
      JSON.stringify({
        success: false,
        error: "Python script failed",
        details: errText,
      }),
      { status: 500, headers: JSON_HEADERS },
    );
  }

  /*const outText = new TextDecoder().decode(stdout).trim();
  let data: unknown;
  try {
    data = JSON.parse(outText);
  } catch {
    console.error("Invalid JSON from python:", outText);
    return new Response(
      JSON.stringify({
        success: false,
        error: "Invalid JSON from python",
        raw: outText,
      }),
      { status: 500, headers: JSON_HEADERS },
    );
  }
*/
const outputText = new TextDecoder().decode(stdout);

// استخراج آخر سطر JSON فقط
const jsonLines = outputText
  .split("\n")
  .map(line => line.trim())
  .filter(line => line.startsWith("{") && line.endsWith("}"));

if (jsonLines.length === 0) {
  console.error("No JSON found in python output:\n", outputText);
  return new Response(
    JSON.stringify({
      success: false,
      error: "No JSON found in python output",
    }),
    { status: 500, headers: JSON_HEADERS },
  );
}

const data = JSON.parse(jsonLines[jsonLines.length - 1]);


  return new Response(JSON.stringify({ success: true, data }), {
    status: 200,
    headers: JSON_HEADERS,
  });
}
async function handleMedia(pathname: string): Promise<Response> {
  const name = decodeURIComponent(pathname.replace("/media/", ""));
  const filePath = join(UPLOAD_DIR, name);

  try {
    const file = await Deno.readFile(filePath);
    // نفترض JPG
    return new Response(file, {
      status: 200,
      headers: {
        "Content-Type": "image/jpeg",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch {
    return new Response("Not found", { status: 404 });
  }
}

const handler = (req: Request): Promise<Response> | Response => {
  const url = new URL(req.url);

  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }

  if (url.pathname === "/health" && req.method === "GET") {
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: JSON_HEADERS,
    });
  }

  if (url.pathname === "/detect" && req.method === "POST") {
    return handleDetect(req);
  }

  if (url.pathname.startsWith("/media/") && req.method === "GET") {
    return handleMedia(url.pathname);
  }

  return new Response("Not found", { status: 404, headers: CORS_HEADERS });
};

console.log(`🚀 ALPR backend listening on http://localhost:${PORT}`);
await serve(handler, { port: PORT });

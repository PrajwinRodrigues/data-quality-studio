// frontend/src/api.js
import axios from "axios";

export const API_BASE ="https://dataqualitystudio.onrender.com";


const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

// helper to inject token if present later (optional)
client.interceptors.request.use((config) => {
  const token = localStorage.getItem("dq_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Upload CSV file -> /upload-csv accepts form field "file"
export async function uploadCsv(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await client.post("/upload-csv", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

// Preview rule -> /preview-rule expects form "file" and form "rule" (json string or simple op)
export async function previewRule(file, ruleObj) {
  const fd = new FormData();
  if (file instanceof File) fd.append("file", file);
  const ruleStr =
    typeof ruleObj === "string" ? ruleObj : JSON.stringify(ruleObj || "");
  fd.append("rule", ruleStr);
  const res = await client.post("/preview-rule", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

// Suggest rules -> /suggest-rules expects "file"
export async function suggestRules(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await client.post("/suggest-rules", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

// Replace NaNs -> /replace-nans expects "file" and form field "strategy"
export async function replaceNaNs(file, strategy = "auto") {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("strategy", strategy);
  const res = await client.post("/replace-nans", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

// REAL auth: register + login

export async function register(email, password, name) {
  const res = await client.post("/auth/register", {
    email,
    password,
    name,
  });
  return res.data; // { access_token, token_type, user }
}

export async function login(email, password) {
  const res = await client.post("/auth/login", {
    email,
    password,
  });
  return res.data; // { access_token, token_type, user }
}

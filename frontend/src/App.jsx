// frontend/src/App.jsx
import React, { useState } from "react";
import "./App.css";
import { uploadCsv, previewRule, suggestRules, replaceNaNs, login, register } from "./api";
import DataActions from "./DataActions";
import UploadZone from "./UploadZone";
import RuleCard from "./RuleCard";
import Sidebar from "./Sidebar";

function ColumnStat({ name, stat = {} }) {
  return (
    <div className="col-stat">
      <h4>{name}</h4>
      {stat.error ? (
        <div className="small">Error: {stat.error}</div>
      ) : (
        <>
          <div className="small">Type: {stat.dtype}</div>
          <div className="small">Missing: {stat.missing_count}</div>
          <div className="small">Unique: {stat.unique_count}</div>
          <ul className="top-list">
            {(stat.top_values && Object.entries(stat.top_values))?.slice(0, 5).map(
              ([v, c]) => (
                <li key={v}>
                  <strong>{v}</strong> — {c}
                </li>
              )
            )}
          </ul>
        </>
      )}
    </div>
  );
}

export default function App() {
  // sidebar section
  const [section, setSection] = useState("upload");

  // auth state
  const [showLogin, setShowLogin] = useState(false);
  const [authLoading, setAuthLoading] = useState(false);
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [loginName, setLoginName] = useState("");
  const [isRegisterMode, setIsRegisterMode] = useState(false);
  const [token, setToken] = useState(localStorage.getItem("dq_token") || null);
  const [currentUser, setCurrentUser] = useState(
    localStorage.getItem("dq_user") || null
  );

  // data state
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [selectedColumn, setSelectedColumn] = useState("");
  const [rule, setRule] = useState("parse_numeric");
  const [before, setBefore] = useState([]);
  const [after, setAfter] = useState([]);
  const [summary, setSummary] = useState({});
  const [nRowsTotal, setNRowsTotal] = useState(null);
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState({});
  const [savedPath, setSavedPath] = useState(null); // <-- NEW

  // AUTH HANDLERS
  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    setAuthLoading(true);
    try {
      let data;
      if (isRegisterMode) {
        data = await register(loginEmail, loginPassword, loginName || "User");
      } else {
        data = await login(loginEmail, loginPassword);
      }

      setToken(data.access_token);
      const displayName = data.user?.name || data.user?.email || "User";
      setCurrentUser(displayName);

      localStorage.setItem("dq_token", data.access_token);
      localStorage.setItem("dq_user", displayName);

      setShowLogin(false);
      setLoginEmail("");
      setLoginPassword("");
      setLoginName("");
      setIsRegisterMode(false);
      alert(
        isRegisterMode
          ? `Account created. Logged in as ${displayName}.`
          : `Logged in as ${displayName}`
      );
    } catch (err) {
      console.error(err);
      alert("Auth failed: " + (err?.response?.data?.detail || err.message));
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    setToken(null);
    setCurrentUser(null);
    localStorage.removeItem("dq_token");
    localStorage.removeItem("dq_user");
  };

  // Upload CSV
  const handleUpload = async () => {
    if (!file) {
      alert("Please select a CSV file first.");
      return;
    }
    setLoading(true);
    try {
      const data = await uploadCsv(file);
      setColumns(data.columns || []);
      setSummary(data.summary || {});
      setNRowsTotal(data.n_rows_total ?? null);
      setBefore(data.preview || []);
      setAfter([]);
      setSuggestions({});
      setSavedPath(data.saved_path || null); // <-- NEW
      alert("File uploaded successfully!");
      // once uploaded, jump to cleaning tools
      setSection("clean");
    } catch (err) {
      console.error(err);
      alert("Upload failed: " + (err?.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Preview rule (calls backend preview-rule)
  // UPDATED to accept full custom payloads,
  // so we can call handlePreviewRule({ op: "scale_numeric" }) for all numeric columns.
  const handlePreviewRule = async (ruleObj = null) => {
    if (!file) {
      alert("Upload a CSV first!");
      return;
    }

    let rulePayload;

    if (ruleObj) {
      // Caller gives the full payload (used by suggestions and "Scale All Numeric" button)
      if (!ruleObj.op) {
        alert("Operation is missing in rule payload!");
        return;
      }
      rulePayload = ruleObj;
    } else {
      // Normal path from Apply Rule UI (single column + selected rule)
      if (!rule) {
        alert("Select an operation first!");
        return;
      }
      if (!selectedColumn) {
        alert("Select a column first!");
        return;
      }
      rulePayload = { op: rule, col: selectedColumn };
    }

    setLoading(true);
    try {
      const data = await previewRule(file, rulePayload, savedPath); // <-- NEW (pass savedPath)
      setBefore(data.before_preview || []);
      setAfter(data.after_preview || []);
      if (data.saved_path) setSavedPath(data.saved_path); // <-- NEW (update savedPath)
      setSection("preview");
    } catch (err) {
      console.error(err);
      alert("Rule preview failed: " + (err?.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Suggest rules
  const handleSuggest = async () => {
    if (!file) {
      alert("Upload a CSV first!");
      return;
    }
    setLoading(true);
    try {
      const data = await suggestRules(file);
      setSuggestions(data.suggestions || {});
      alert("Suggestions ready — check the Suggestions panel.");
      setSection("clean");
    } catch (err) {
      console.error(err);
      alert("Suggest failed: " + (err?.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Replace NaNs (auto or zero)
  const handleReplaceNaNs = async (mode = "auto") => {
    if (!file) {
      alert("Upload a CSV first!");
      return;
    }
    setLoading(true);
    try {
      const data = await replaceNaNs(file, mode, savedPath); // <-- NEW (pass savedPath)
      if (data.preview) setAfter(data.preview);
      if (data.saved_path) setSavedPath(data.saved_path); // <-- NEW (update savedPath)
      alert("Replace NaNs completed (preview available).");
      setSection("preview");
    } catch (err) {
      console.error(err);
      alert("Replace NaNs failed: " + (err?.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Refresh summary/preview by re-uploading
  const refreshPreview = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const data = await uploadCsv(file);
      setColumns(data.columns || []);
      setSummary(data.summary || {});
      setNRowsTotal(data.n_rows_total ?? null);
      setBefore(data.preview || []);
      setSavedPath(data.saved_path || null); // <-- NEW
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Download cleaned CSV (from `after` preview)
  const handleDownload = () => {
    if (!after || !after.length) {
      alert("No cleaned data to download!");
      return;
    }
    const csvHeader = Object.keys(after[0]).join(",");
    const csvRows = after.map((row) =>
      Object.values(row)
        .map((v) => (v === null || v === undefined ? "" : String(v)))
        .join(",")
    );
    const csvContent = [csvHeader, ...csvRows].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cleaned_data.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ display: "flex", background: "#020617", minHeight: "100vh" }}>
      <Sidebar active={section} onChange={setSection} />

      <div
        className="app"
        style={{
          marginLeft: 240,
          padding: 24,
          color: "#ddd",
          background: "#111827",
          minHeight: "100vh",
          flex: 1,
        }}
      >
        {/* Top navbar */}
        <header className="topbar">
          <div className="topbar-left">
            <span className="logo-dot" />
            <span className="topbar-title">Data Quality Studio</span>
          </div>
          <div className="topbar-right">
            {currentUser ? (
              <>
                <span className="user-chip">Hi, {currentUser}</span>
                <button onClick={handleLogout}>Logout</button>
              </>
            ) : (
              <button onClick={() => setShowLogin(true)}>Login</button>
            )}
          </div>
        </header>

        <h1 style={{ marginBottom: 8, marginTop: 16 }}>
          🧹 Data Quality Studio — Enhanced Demo
        </h1>
        <p style={{ marginTop: 0, opacity: 0.8 }}>
          Upload a CSV, inspect data quality, apply cleaning rules and download the
          cleaned file.
        </p>

        {/* ====== SECTION: UPLOAD ====== */}
        {section === "upload" && (
          <section className="panel" style={{ marginTop: 16 }}>
            <div className="panel-header">
              <h2>1. Upload CSV</h2>
              <span className="panel-tag">Required</span>
            </div>

            <UploadZone onFileSelect={setFile} />

            <div className="upload-actions">
              <button onClick={handleUpload} disabled={loading || !file}>
                {loading ? "Working..." : "Upload & Analyze"}
              </button>
              <button onClick={handleSuggest} disabled={loading || !file}>
                {loading ? "Working..." : "Suggest Rules"}
              </button>
              <button onClick={() => handleReplaceNaNs("auto")} disabled={loading || !file}>
                Auto Fill NaNs
              </button>
              <button onClick={() => handleReplaceNaNs("zero")} disabled={loading || !file}>
                Fill NaNs with 0
              </button>
            </div>

            {nRowsTotal !== null && (
              <div className="dataset-summary-row">
                <strong>Total rows:</strong> {nRowsTotal} &nbsp; • &nbsp;
                <strong>Columns:</strong> {columns.length}
              </div>
            )}
          </section>
        )}

        {/* ====== SECTION: CLEANING TOOLS ====== */}
        {section === "clean" && (
          <>
            {columns.length === 0 ? (
              <div style={{ marginTop: 24 }} className="small">
                No dataset loaded. Go to <strong>Dataset Upload</strong> and upload a CSV
                first.
              </div>
            ) : (
              <>
                {/* Apply rule */}
                <section className="panel" style={{ marginTop: 18 }}>
                  <h3>2. Apply Rule</h3>
                  <div className="apply-row">
                    <label>
                      Column:
                      <select
                        value={selectedColumn}
                        onChange={(e) => setSelectedColumn(e.target.value)}
                        style={{ marginLeft: 8 }}
                      >
                        <option value="">-- Select Column --</option>
                        {columns.map((col) => (
                          <option key={col} value={col}>
                            {col}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label style={{ marginLeft: 12 }}>
                      Rule:
                      <select
                        value={rule}
                        onChange={(e) => setRule(e.target.value)}
                        style={{ marginLeft: 8 }}
                      >
                        <option value="parse_numeric">Parse Numeric</option>
                        <option value="strip_spaces">Strip Spaces</option>
                        <option value="lowercase">Lowercase</option>
                        <option value="parse_date">Parse Date</option>
                        <option value="fill_missing">Fill Missing</option>
                        <option value="dedupe_by_cols">Dedupe Rows</option>
                        <option value="scale_numeric">Scale Numeric (standardize)</option>
                      </select>
                    </label>

                    {/* Standard single-column preview */}
                    <button
                      onClick={() => handlePreviewRule()}
                      disabled={loading || !selectedColumn}
                      style={{ marginLeft: 12 }}
                    >
                      {loading ? "Processing..." : "Preview Rule"}
                    </button>

                    {/* NEW: scale ALL numeric columns */}
                    <button
                      onClick={() => {
                        setRule("scale_numeric");
                        setSelectedColumn("");
                        handlePreviewRule({ op: "scale_numeric" });
                      }}
                      disabled={loading}
                      style={{ marginLeft: 12 }}
                    >
                      {loading ? "Processing..." : "Scale All Numeric Columns"}
                    </button>
                  </div>
                </section>

                {/* Data summary */}
                <section className="panel" style={{ marginTop: 18 }}>
                  <h3>3. Data Summary</h3>
                  <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                    {columns.map((col) => (
                      <ColumnStat key={col} name={col} stat={summary[col] || {}} />
                    ))}
                  </div>
                </section>

                {/* Suggestions */}
                <section className="panel" style={{ marginTop: 18 }}>
                  <h3>4. Suggestions</h3>
                  <div className="rules-list">
                    {Object.keys(suggestions).length === 0 && (
                      <div className="small">
                        No suggestions yet. Click <strong>Suggest Rules</strong> in the
                        upload section.
                      </div>
                    )}
                    {Object.entries(suggestions).map(([col, rules]) =>
                      rules.map((s, i) => (
                        <RuleCard
                          key={`${col}-${i}`}
                          col={col}
                          op={s.op}
                          reason={s.reason}
                          onApply={() => handlePreviewRule({ op: s.op, col })}
                        />
                      ))
                    )}
                  </div>
                </section>
              </>
            )}
          </>
        )}

        {/* ====== SECTION: PREVIEW ====== */}
        {section === "preview" && (
          <>
            {before.length === 0 ? (
              <div style={{ marginTop: 24 }} className="small">
                No preview yet. Apply a rule or run a NaN replacement from{" "}
                <strong>Cleaning Tools</strong>.
              </div>
            ) : (
              <section className="panel" style={{ marginTop: 24 }}>
                <h2>5. Preview Rule Result</h2>
                <p className="small">
                  Current rule:&nbsp;
                  <code>{JSON.stringify({ op: rule, col: selectedColumn })}</code>
                </p>
                <div style={{ display: "flex", gap: 20, marginTop: 10 }}>
                  <div style={{ flex: 1 }}>
                    <h3>Before</h3>
                    <TableFromRows rows={before} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <h3>After (changes highlighted)</h3>
                    <DiffTable beforeRows={before} afterRows={after} />
                  </div>
                </div>
                <div style={{ marginTop: 12 }}>
                  <button onClick={handleDownload} disabled={!after.length}>
                    ⬇️ Download Cleaned CSV
                  </button>
                </div>
              </section>
            )}

            {/* DataActions area always useful in Preview tab */}
            <section style={{ marginTop: 28, borderTop: "1px solid #333", paddingTop: 18 }}>
              <DataActions file={file} refreshPreview={refreshPreview} onPreviewResult={setAfter} />
            </section>
          </>
        )}

        {/* ====== SECTION: HISTORY (placeholder) ====== */}
        {section === "history" && (
          <section className="panel" style={{ marginTop: 24 }}>
            <h2>Cleaning History</h2>
            <p className="small">
              Coming soon: a timeline of datasets and rules you've applied.
              For now, use this space in the demo to talk about how you could store
              runs in a database (user, timestamp, file name, rules applied, etc.).
            </p>
          </section>
        )}

        {/* Login modal */}
        {showLogin && (
          <div className="login-backdrop">
            <div className="login-card">
              <h3>{isRegisterMode ? "Create Account" : "Login"}</h3>
              <p className="small">
                {isRegisterMode
                  ? "Sign up with your email and a password."
                  : "Login with the account you created."}
              </p>
              <form onSubmit={handleLoginSubmit} className="login-form">
                {isRegisterMode && (
                  <label>
                    Name
                    <input
                      type="text"
                      value={loginName}
                      onChange={(e) => setLoginName(e.target.value)}
                      placeholder="Prajwin Rodrigues"
                      required
                    />
                  </label>
                )}
                <label>
                  Email
                  <input
                    type="email"
                    value={loginEmail}
                    onChange={(e) => setLoginEmail(e.target.value)}
                    required
                  />
                </label>
                <label>
                  Password
                  <input
                    type="password"
                    value={loginPassword}
                    onChange={(e) => setLoginPassword(e.target.value)}
                    required
                  />
                </label>
                <div className="login-actions">
                  <button
                    type="button"
                    onClick={() => setShowLogin(false)}
                    disabled={authLoading}
                  >
                    Cancel
                  </button>
                  <button type="submit" disabled={authLoading}>
                    {authLoading
                      ? isRegisterMode
                        ? "Creating..."
                        : "Logging in..."
                      : isRegisterMode
                      ? "Sign up"
                      : "Login"}
                  </button>
                </div>
              </form>
              <p className="small" style={{ marginTop: 8 }}>
                {isRegisterMode ? "Already have an account? " : "Need an account? "}
                <button
                  type="button"
                  style={{
                    background: "none",
                    border: "none",
                    color: "#38bdf8",
                    cursor: "pointer",
                    padding: 0,
                  }}
                  onClick={() => setIsRegisterMode((v) => !v)}
                >
                  {isRegisterMode ? "Log in" : "Sign up"}
                </button>
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* Small helper table renderer */
function TableFromRows({ rows = [] }) {
  if (!rows || !rows.length) return <div>No preview yet</div>;
  const cols = Object.keys(rows[0] || {});
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", color: "#eee" }}>
      <thead>
        <tr>
          {cols.map((c) => (
            <th
              key={c}
              style={{ border: "1px solid #333", padding: 6, background: "#020617" }}
            >
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            {cols.map((c, j) => (
              <td key={j} style={{ border: "1px solid #333", padding: 6 }}>
                {String(r[c] ?? "")}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* After table with diff highlighting */
function DiffTable({ beforeRows = [], afterRows = [] }) {
  if (!afterRows || !afterRows.length) return <div>No preview yet</div>;
  const cols = Object.keys(afterRows[0] || {});

  return (
    <table style={{ width: "100%", borderCollapse: "collapse", color: "#eee" }}>
      <thead>
        <tr>
          {cols.map((c) => (
            <th
              key={c}
              style={{ border: "1px solid #333", padding: 6, background: "#020617" }}
            >
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {afterRows.map((row, i) => (
          <tr key={i}>
            {cols.map((c, j) => {
              const beforeVal = beforeRows[i]?.[c];
              const afterVal = row[c];
              const beforeStr =
                beforeVal === null || beforeVal === undefined ? "" : String(beforeVal);
              const afterStr =
                afterVal === null || afterVal === undefined ? "" : String(afterVal);
              const changed = beforeStr !== afterStr;

              return (
                <td
                  key={j}
                  style={{
                    border: "1px solid #333",
                    padding: 6,
                    background: changed ? "#064e3b" : "transparent",
                    color: changed ? "#bbf7d0" : "#e5e7eb",
                    transition: "background 0.15s ease",
                  }}
                >
                  {afterStr}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

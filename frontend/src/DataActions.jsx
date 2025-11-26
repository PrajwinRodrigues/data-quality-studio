// frontend/src/DataActions.jsx
import React, { useState } from "react";
import { suggestRules, previewRule, replaceNaNs } from "./api";

/**
 * DataActions component
 * Props:
 *  - file: the uploaded File object
 *  - refreshPreview: function to re-run uploadCsv (so App.jsx refreshes summary/preview)
 *  - onPreviewResult(afterPreview): optional callback to set the "after" preview in parent
 */
export default function DataActions({ file, refreshPreview, onPreviewResult }) {
  const [loadingApplyAll, setLoadingApplyAll] = useState(false);
  const [loadingReplace, setLoadingReplace] = useState(false);
  const [msg, setMsg] = useState(null);

  // apply all suggestions at a preview level:
  // we call suggestRules(file) then preview the first suggestion found
  async function applyAllSuggestions() {
    setMsg(null);
    if (!file) {
      setMsg("Upload a CSV first.");
      return;
    }
    setLoadingApplyAll(true);
    try {
      const sugResp = await suggestRules(file);
      const suggestions = sugResp?.suggestions || {};
      // flatten to first suggestion per column
      const first = Object.entries(suggestions).flatMap(([col, arr]) =>
        (arr && arr.length) ? [{ col, op: arr[0].op, reason: arr[0].reason }] : []
      )[0];

      if (!first) {
        setMsg("No suggestions available to apply.");
        setLoadingApplyAll(false);
        return;
      }

      // preview applying the first suggestion
      const ruleObj = { op: first.op, col: first.col };
      const previewResp = await previewRule(file, ruleObj);
      if (previewResp?.after_preview && typeof onPreviewResult === "function") {
        onPreviewResult(previewResp.after_preview);
      }
      setMsg(`Previewed suggestion ${first.op} on ${first.col}`);
      // optionally refresh main preview/summary (re-upload)
      if (typeof refreshPreview === "function") refreshPreview();
    } catch (err) {
      console.error(err);
      setMsg("Failed to apply suggestions: " + (err?.response?.data?.detail || err.message));
    } finally {
      setLoadingApplyAll(false);
    }
  }

  async function replaceNans(strategy = "auto") {
    setMsg(null);
    if (!file) {
      setMsg("Upload a CSV first.");
      return;
    }
    setLoadingReplace(true);
    try {
      const data = await replaceNaNs(file, strategy);
      // backend returns preview in { preview: [...] }
      if (data?.preview && typeof onPreviewResult === "function") {
        onPreviewResult(data.preview);
      }
      setMsg("NaN replacement completed (preview available).");
      if (typeof refreshPreview === "function") refreshPreview();
    } catch (err) {
      console.error(err);
      setMsg("Failed to replace NaNs: " + (err?.response?.data?.detail || err.message));
    } finally {
      setLoadingReplace(false);
    }
  }

  return (
    <div style={{ marginTop: 30 }}>
      <h3>⚙️ Data Cleaning Actions</h3>
      <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
        <button onClick={applyAllSuggestions} disabled={loadingApplyAll}>
          {loadingApplyAll ? "Applying..." : "Apply All Suggestions"}
        </button>

        <button onClick={() => replaceNans("auto")} disabled={loadingReplace}>
          {loadingReplace ? "Replacing..." : "Replace NaNs (auto)"}
        </button>

        <button onClick={() => replaceNans("zero")} disabled={loadingReplace}>
          Replace NaNs (zero)
        </button>
      </div>

      {msg && <div style={{ marginTop: "10px", color: "#ffb", fontFamily: "monospace" }}>{msg}</div>}
    </div>
  );
}

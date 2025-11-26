// frontend/src/Sidebar.jsx
import React from "react";
import { FileSpreadsheet, Settings2, FlaskConical, BarChart3 } from "lucide-react";

export default function Sidebar({ active, onChange }) {
  const items = [
    { id: "upload", label: "Dataset Upload", icon: <FileSpreadsheet size={20} /> },
    { id: "clean", label: "Cleaning Tools", icon: <FlaskConical size={20} /> },
    { id: "preview", label: "Preview", icon: <Settings2 size={20} /> },
    { id: "history", label: "History", icon: <BarChart3 size={20} /> },
  ];

  return (
    <div style={{
      width: 220,
      background: "#0f172a",
      borderRight: "1px solid #1e293b",
      padding: "16px",
      height: "100vh",
      position: "fixed",
      left: 0,
      top: 0,
      color: "#eee"
    }}>
      <h2 style={{ fontSize: 20, marginBottom: 20 }}>📊 DQ Studio</h2>

      {items.map(item => (
        <div
          key={item.id}
          onClick={() => onChange(item.id)}
          style={{
            display: "flex",
            gap: 12,
            alignItems: "center",
            padding: "10px 12px",
            borderRadius: 6,
            cursor: "pointer",
            background: active === item.id ? "#1e293b" : "transparent",
            color: active === item.id ? "#22d3ee" : "#cbd5e1",
            marginBottom: 6
          }}
        >
          {item.icon} {item.label}
        </div>
      ))}
    </div>
  );
}

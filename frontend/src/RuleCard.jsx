// src/RuleCard.jsx
import { Wand2 } from "lucide-react";

export default function RuleCard({ col, op, reason, onApply }) {
  return (
    <div className="rule-card">
      <div>
        <div className="rule-title">
          <Wand2 size={18} />
          <span>{op}</span>
          <span className="rule-col">· {col}</span>
        </div>
        <div className="rule-reason">{reason}</div>
      </div>
      <button className="rule-apply-btn" onClick={onApply}>
        Apply
      </button>
    </div>
  );
}

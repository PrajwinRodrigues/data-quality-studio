// src/UploadZone.jsx
import { useDropzone } from "react-dropzone";

export default function UploadZone({ onFileSelect }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "text/csv": [".csv"] },
    onDrop: (acceptedFiles) => {
      if (acceptedFiles && acceptedFiles[0]) {
        onFileSelect(acceptedFiles[0]);
      }
    },
  });

  return (
    <div
      {...getRootProps()}
      className="upload-zone"
    >
      <input {...getInputProps()} />
      <p className="upload-zone-text">
        {isDragActive ? "Drop CSV here..." : "Drag & drop CSV or click to browse"}
      </p>
      <p className="upload-zone-sub">
        Max a few MB · .csv only
      </p>
    </div>
  );
}

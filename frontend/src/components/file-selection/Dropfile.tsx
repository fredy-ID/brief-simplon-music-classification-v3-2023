import { createSignal } from "solid-js";
import { file, setFile } from "./FileSelection";

export default function() {

  const handleDrop = (e: any) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];

    if (droppedFile && (droppedFile.type === "audio/mp3" || droppedFile.type === "audio/wav")) {
      setFile(droppedFile);
    } else {
      alert("Veuillez sélectionner un fichier MP3 valide.");
    }
  };

  const handleDragOver = (e: any) => {
    e.preventDefault();
  };

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      style={{
        width: "300px",
        height: "100px",
        border: "2px dashed #ccc",
        display: "flex",
        "align-items": "center",
        "justify-content": "center",
        cursor: "pointer",
      }}
    >
      {file() ? (
        <div>
          <p>Fichier sélectionné : {file()?.name}</p>
          <audio controls src={URL.createObjectURL(file() as File)}></audio>
        </div>
      ) : (
        <p>Glissez-déposez un fichier MP3 ici</p>
      )}
    </div>
  );
}

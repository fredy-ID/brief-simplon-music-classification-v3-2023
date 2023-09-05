import { createEffect, createSignal } from "solid-js";

export  const [file, setFile] = createSignal<File>();

export default function(props: {onSelect: (e: any) => void}){
    const handleFileSelect = (e: any) => {
      setFile(e.target.files[0])
    };

    createEffect(() => {
      console.log()
    })
  
    return (
      <div class="relative inline-block">
        <label
          for="file-input"
          class="px-4 py-2 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600"
        >
          Sélectionner un fichier
        </label>
        <input
          id="file-input"
          type="file"
          class="hidden"
          onChange={handleFileSelect}
        />
      </div>
    );
}
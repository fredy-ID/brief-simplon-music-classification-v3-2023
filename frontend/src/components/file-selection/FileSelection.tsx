import { createSignal } from "solid-js";
import Dropfile from "./Dropfile";

export const [file, setFile] = createSignal<File | undefined>();
export const getFileBlobUrl = () => BlobFileUrl

let BlobFileUrl: string;


export default function(props: {onSelect: (e: any) => void}){
    const handleFileSelect = (e: any) => {
      setFile(e.target.files[0])
    };
 
    return (
      <div class="w-full">
        <label
          for="file-input"
          class="px-3 text-sm py-2 text-center w-full bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600 block "
        >
          Sélectionner un fichier
        </label>
        <input
          id="file-input"
          type="file"
          class="hidden"
          onChange={handleFileSelect}
        />

        <Dropfile />
      </div>
    );
}
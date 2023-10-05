import { createSignal } from "solid-js";
import Dropfile from "./Dropfile";

export const [file, setFile] = createSignal<File | undefined>();
export const [modele, setModele] = createSignal('1');

export const getFileBlobUrl = () => BlobFileUrl

let BlobFileUrl: string;

const handleModeleChange = (e: Event) => {
  const target = e.target as HTMLSelectElement;
  setModele(target.value);
  console.log(modele())
};


export default function(props: {onSelect: (e: any) => void}){
    const handleFileSelect = (e: any) => {
      setFile(e.target.files[0])
    };
 
    return (
      <div class="w-full">
        <div class="my-5">
            <label class="modele">Modèle :</label>
            <select id="modele" value={modele()} onChange={handleModeleChange}>
                <option value="1">1</option>
                <option value="2">2</option>
            </select>
        </div>
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
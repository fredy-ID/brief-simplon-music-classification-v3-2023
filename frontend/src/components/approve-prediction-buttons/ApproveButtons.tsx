import { Show, createSignal } from 'solid-js';
import "./ApproveButtons.css"
import { setFile } from '../file-selection/FileSelection';
import { setResponse } from '../results/Result';

export const [goodResponse, setGoodResponse] = createSignal<boolean>(true)

export default function () {
 

  const genderChoicesButton = () => {
    return 
  }

  const onClickGoodResponse = () => { 
    setFile(undefined)
    setResponse(undefined)
  }

  return (
    <Show when={goodResponse()}>
      <div class="flex gap-2">
        <button class='bg-red-500 px-4 py-2' onClick={() => setGoodResponse(false)}>Mauvaise réponse</button>
        <button class='bg-green-500 px-4 py-2' onClick={onClickGoodResponse}>Bonne réponse</button>
      </div>
    </Show>
  );
}

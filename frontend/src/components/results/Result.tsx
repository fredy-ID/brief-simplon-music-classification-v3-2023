import { Show, createEffect, createSignal } from "solid-js";
import { file, modele } from "../file-selection/FileSelection";
import Api from "../../services/api.service";

interface InterfaceResponse {
    id: number,
    msg: string,
    predicted_classes: string | string[],
    prediction_scores: number[];
}

export const [response, setResponse] = createSignal<InterfaceResponse>()


export default function () {
    
    
    let loader = false

    createEffect( async () => {
        if(file() != undefined){
            const formData = new FormData()
            formData.append('music', file() as File)
            formData.append('modele', modele())
            console.log("send request post to predict");
            console.log(modele())
            loader = true
            const json = await Api.post("/predict/", formData)
            console.log("response json", json);
            setResponse(json)
        }
    })
    
    const LoaderSpin = () => {
        return <svg class="animate-spin" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path opacity="0.2" fill-rule="evenodd" clip-rule="evenodd" d="M12 19C12.9193 19 13.8295 18.8189 14.6788 18.4672C15.5281 18.1154 16.2997 17.5998 16.9497 16.9497C17.5998 16.2997 18.1154 15.5281 18.4672 14.6788C18.8189 13.8295 19 12.9193 19 12C19 11.0807 18.8189 10.1705 18.4672 9.32122C18.1154 8.47194 17.5998 7.70026 16.9497 7.05025C16.2997 6.40024 15.5281 5.88463 14.6788 5.53284C13.8295 5.18106 12.9193 5 12 5C10.1435 5 8.36301 5.7375 7.05025 7.05025C5.7375 8.36301 5 10.1435 5 12C5 13.8565 5.7375 15.637 7.05025 16.9497C8.36301 18.2625 10.1435 19 12 19ZM12 22C17.523 22 22 17.523 22 12C22 6.477 17.523 2 12 2C6.477 2 2 6.477 2 12C2 17.523 6.477 22 12 22Z" fill="#2253FF"/>
        <path d="M2 12C2 6.477 6.477 2 12 2V5C10.1435 5 8.36301 5.7375 7.05025 7.05025C5.7375 8.36301 5 10.1435 5 12H2Z" fill="#2253FF"/>
        </svg>
        
    }



    return <div id="result">
        <Show when={response() != undefined} fallback={<LoaderSpin />}>
            <p>{response()?.msg}</p>
            <p>{response()?.predicted_classes}</p>
        </Show>
    </div>
}
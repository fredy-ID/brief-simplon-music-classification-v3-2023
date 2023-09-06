import { Show, createEffect } from "solid-js";
import { file } from "../file-selection/FileSelection";
import Api from "../../services/api.service";

export default function () {
    let loader = false

    createEffect( async () => {
        if(file() != undefined){
            const formData = new FormData()
            formData.append('music', file() as File)
            console.log("send request post to predict");
            loader = true
            const json = await Api.post("/predict/", formData)
            console.log("response json", json);
        }
    })
    
    return <div id="result">
        <Show when={!loader} fallback={"true"}>
            false
        </Show>
    </div>
}
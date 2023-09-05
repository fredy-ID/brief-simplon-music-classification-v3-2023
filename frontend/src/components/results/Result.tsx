import { createEffect } from "solid-js";
import { file } from "../file-selection/FileSelection";
import Api from "../../services/api.service";

export default function () {
    
    createEffect( async () => {
        if(file() != undefined){
            const json = Api.post("", {})
            console.log(json);
            
        }
    })
    
    return <div id="result">
        test
    </div>
}
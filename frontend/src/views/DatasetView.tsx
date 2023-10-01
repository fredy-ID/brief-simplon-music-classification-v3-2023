import { createEffect, createSignal, onMount } from "solid-js";
import { GENRE_CHOICES } from "../contant";
import "./DatasetView.css"
import Api from "../services/api.service";

export default function(){
    const [response, setResponse] = createSignal<any>(undefined)

    const canTrain = () => {

        if(Object.keys(response()).length != GENRE_CHOICES.length){
            return
        }        
    }

    onMount( async () => {
        setResponse(await Api.get('/predictions/'))
        console.log(response()['rock']);
        console.log(response());
        canTrain()
    })

    return <section>
        <div class="genders-registered-data">
            {GENRE_CHOICES.map((genre, index) => (
                <div class="block">
                    <p class="gender-label">{genre}</p>
                    <p class="gender-label value">{response() != undefined &&  response()[genre] != undefined ? response()[genre] : 0 }</p>
                </div>
            ))}
        </div>
    </section>
}
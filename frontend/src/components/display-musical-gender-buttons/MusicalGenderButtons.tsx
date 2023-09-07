import { Show, createSignal } from "solid-js";
import Api from "../../services/api.service";
import { response } from "../results/Result";
import { GENRE_CHOICES } from "../../contant";


export default function (){
    const [feedbackResponse, setFeedbackResponse] = createSignal<{msg: string}>()

    const onClick = async (genre: string) => {
        const json = await Api.post('/feedback/' + response()?.id + "/", {genre_feedback: genre}, true)
        if(json == false){
            setFeedbackResponse({
                msg: "Désoler une erreur est survenue !"
            })
            return
        }
        setFeedbackResponse(json)
    }

    return <div id="audi-card" class="flex justify-center flex-wrap">
        <Show when={feedbackResponse() == undefined} fallback={feedbackResponse()?.msg}>

            {GENRE_CHOICES.map((genre, index) => (
                <button onClick={() => onClick(genre)}>{genre}</button>
            ))}
        </Show>
        </div>
}   
import { Show } from "solid-js";
import { goodResponse } from "../approve-prediction-buttons/ApproveButtons";
import Api from "../../services/api.service";
import { response } from "../results/Result";

export default function (){
    const GENRE_CHOICES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'];
    const onClick = async (genre: string) => {
        console.log(genre);   
        console.log(response());
        
        const json = await Api.post('/feedback/' + response()?.id + "/", {}, true)
        console.log(json);
        
    }
    return <div id="audi-card" class="flex flex-wrap">
            {GENRE_CHOICES.map((genre, index) => (
            <button onClick={() => onClick(genre)}>{genre}</button>
            ))}
        </div>
}   
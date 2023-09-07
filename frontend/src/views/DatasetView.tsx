import { GENRE_CHOICES } from "../contant";
import "./DatasetView.css"

export default function(){
    return <section>
        <div class="genders-registered-data">

        {GENRE_CHOICES.map((genre, index) => (
            <div class="block">
                <p class="gender-label">{genre}</p>
                <p class="gender-label value">1</p>
            </div>
            ))}
            </div>
    </section>
}
import { createEffect, createSignal } from "solid-js"
import "./PanelsButton.css"

export enum Panels {
    prediction = "prediction",
    data = "data",
    train = "train"
}

export const [onPanel, setOnPanel] = createSignal<Panels>(Panels.prediction)

export default function(){

    const onClick = (panel: Panels) => {
        setOnPanel((prev) => prev = panel)
    } 

    return<section class='panels-button'>
        <div onClick={() => onClick(Panels.prediction)} class="btn-panel">
            Prediction
        </div>
        <div onClick={() => onClick(Panels.data)} class="btn-panel">
            Data
        </div>
        <div onClick={() => onClick(Panels.train)} class="btn-panel">
            Entrainement
        </div>
    </section>
}
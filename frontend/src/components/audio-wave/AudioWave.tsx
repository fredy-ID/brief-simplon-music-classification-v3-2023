import { createEffect, createSignal, onMount } from "solid-js";
import { file } from "../file-selection/FileSelection";

import WaveSurfer from "wavesurfer.js"

export const [wavesurfer, setWavesurfer] = createSignal<WaveSurfer>()

export default function (){
    const [refWave, setRefWave] = createSignal<HTMLElement>()

    createEffect(() => {
        if(file() != undefined){
            const blob = new Blob([file() as File], { type: 'audio/mp3' });
            const audioUrlObject = URL.createObjectURL(blob);

            setWavesurfer(WaveSurfer.create({
                container: refWave() as HTMLElement,
                waveColor: 'rgb(200, 0, 200)',
                progressColor: 'rgb(100, 0, 100)',
                url: audioUrlObject,
                height: 20,
                barWidth: 2,
                barGap: 1,
                barRadius: 2,
                autoplay: false
            }))
        }
    })


    return <div ref={setRefWave}></div>
}
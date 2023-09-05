import { createEffect, createSignal, onMount } from "solid-js";
import { file } from "../file-selection/FileSelection";

import WaveSurfer from "wavesurfer.js"

export default function (){
    const [refWave, setRefWave] = createSignal<HTMLElement>()
    const [wavesurfer, setWavesurfer] = createSignal<WaveSurfer>()

    createEffect(() => {
        if(file() != undefined){
            const blob = new Blob([file() as File], { type: 'audio/mp3' });
            const audioUrlObject = URL.createObjectURL(blob);

            setWavesurfer(WaveSurfer.create({
                container: refWave() as HTMLElement,
                waveColor: 'rgb(200, 0, 200)',
                progressColor: 'rgb(100, 0, 100)',
                url: audioUrlObject,
                height: 60,
                barWidth: 2,
                barGap: 1,
                barRadius: 2,
                autoplay: true
            }))
        }
    })


    return <div ref={setRefWave} class="mt-4"></div>
}
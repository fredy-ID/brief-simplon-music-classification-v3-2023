import { Show, createEffect, createSignal } from "solid-js";
import ApproveButtons, { goodResponse } from "../components/approve-prediction-buttons/ApproveButtons";
import AudioWave, { wavesurfer } from "../components/audio-wave/AudioWave";
import MusicalGenderButtons from "../components/display-musical-gender-buttons/MusicalGenderButtons";
import FileSelection, { file } from "../components/file-selection/FileSelection";
import Result, { response } from "../components/results/Result";
import PlaySVG from "../SVGComponents/PlaySVG";
import PauseSVG from "../SVGComponents/PauseSVG";

export default function () {
    const play = () => {
        if (wavesurfer()?.isPlaying()) {
            wavesurfer()?.pause()
        } else
            wavesurfer()?.play()
    }

    const [isPlaying, setIsPlaying] = createSignal(false)

    createEffect(() => {
        if (wavesurfer() != undefined) {
            wavesurfer()?.on("play", () => setIsPlaying(true) )
            wavesurfer()?.on("pause", () => setIsPlaying(false))
        }
    })

    const class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    return <>
        <section class='flex flex-wrap w-full  items-center justify-around '>
            <Show when={file() == undefined}>
                <div class="block w-auto min-w-[150px]">
                    <FileSelection onSelect={(e) => {
                        console.log("test", e);
                    }} />
                </div>
            </Show>

            <Show when={file() != undefined}>
                <div class="w-[78%]">
                    <AudioWave />
                </div>

                <div class="w-[5%] flex items-end cursor-pointer" onClick={play}>
                    <Show when={isPlaying()} fallback={<PlaySVG />}>
                        <PauseSVG />
                    </Show>
                </div>
            </Show>
        </section>

        <Show when={file() != undefined}>
            <section class='results flex justify-center mt-5'>
                <Result />
            </section>
        </Show>

        <Show when={response() != undefined}>
            <section class='results flex flex-col justify-center mt-5'>
                <div class="prediction-scores">
                    {response()?.prediction_scores.map((score, index) => (
                        <div key={index}>
                            {`${class_names[index]}: ${score.toFixed(2)} %`}
                        </div>
                    ))}
                </div>
            </section>
        </Show>

        <Show when={response() != undefined}>
            <section class='results flex justify-center mt-5'>
                <ApproveButtons />
            </section>
        </Show>

        <Show when={!goodResponse()}>
            <MusicalGenderButtons />
        </Show>
    </>
}
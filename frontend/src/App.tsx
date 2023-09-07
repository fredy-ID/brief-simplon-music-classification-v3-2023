import { Show, type Component, createSignal, createEffect, onMount } from 'solid-js';

import styles from './App.module.css';
import TopBar from './components/top-bar/TopBar';
import FileSelection, { file } from './components/file-selection/FileSelection';
import AudioWave, { wavesurfer } from './components/audio-wave/AudioWave';
import AudioCard from './components/audio-card/AudioCard';
import Result, { response } from './components/results/Result';
import ApproveButtons, { goodResponse } from './components/approve-prediction-buttons/ApproveButtons'
import prediction from './components/results/Result';
import MusicalGenderButtons from './components/display-musical-gender-buttons/MusicalGenderButtons';

const PauseButton = () => {
  return <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M2 18C2 19.886 2 20.828 2.586 21.414C3.172 22 4.114 22 6 22C7.886 22 8.828 22 9.414 21.414C10 20.828 10 19.886 10 18V6C10 4.114 10 3.172 9.414 2.586C8.828 2 7.886 2 6 2C4.114 2 3.172 2 2.586 2.586C2 3.172 2 4.114 2 6V14M22 6C22 4.114 22 3.172 21.414 2.586C20.828 2 19.886 2 18 2C16.114 2 15.172 2 14.586 2.586C14 3.172 14 4.114 14 6V18C14 19.886 14 20.828 14.586 21.414C15.172 22 16.114 22 18 22C19.886 22 20.828 22 21.414 21.414C22 20.828 22 19.886 22 18V10" stroke="white" stroke-width="1.5" stroke-linecap="round" />
  </svg>
}

const PlayButton = () => {
  return <svg width="24" height="24" viewBox="0 0 24 24" stroke='red' xmlns="http://www.w3.org/2000/svg">
    <path fill='red' d="M3 12V18.967C3 21.277 5.534 22.736 7.597 21.615L10.8 19.873M3 8V5.033C3 2.723 5.534 1.264 7.597 2.385L20.409 9.353C20.8893 9.60841 21.291 9.98969 21.5712 10.456C21.8514 10.9223 21.9994 11.456 21.9994 12C21.9994 12.544 21.8514 13.0777 21.5712 13.544C21.291 14.0103 20.8893 14.3916 20.409 14.647L14.003 18.131" stroke="white" stroke-width="1.5" stroke-linecap="round" />
  </svg>
}
const App: Component = () => {
  const [isPlaying, setIsPlaying] = createSignal(false)

  createEffect(() => {
    if(wavesurfer() != undefined){
      wavesurfer()?.on("play", () => {
        setIsPlaying(true)
      })

      wavesurfer()?.on("pause", () => {
        setIsPlaying(false)
      })
    }
  })



  const play = () => {
    if(wavesurfer()?.isPlaying()){
      wavesurfer()?.pause()
    }else
      wavesurfer()?.play()
    
  }

  return (
    <div class={styles.App}>
      <TopBar />

      <section id='layout'>
        <section class='flex flex-wrap w-full  items-center justify-around '>
          <div class="block w-auto min-w-[150px]">
            <FileSelection onSelect={(e) => {
              console.log("test", e);
            }} />
          </div>

          <Show when={file() != undefined}>
            <div class="w-[78%]">
              <AudioWave />
            </div>

            <div class="w-[5%] flex items-end cursor-pointer" onClick={play}>
              <Show when={isPlaying()} fallback={<PlayButton />}>
                <PauseButton />
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
          <section class='results flex justify-center mt-5'>
            <ApproveButtons />
          </section>
        </Show>
        
        <Show when={!goodResponse()}>
          <MusicalGenderButtons />
        </Show>
      </section>

    </div>
  );
};

export default App;

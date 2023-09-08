import { Show, type Component, createSignal, createEffect, onMount, Switch, Match } from 'solid-js';

import styles from './App.module.css';
import TopBar from './components/top-bar/TopBar';
import PanelsButton, { Panels, onPanel } from './components/panels-buttons/PanelsButton';
import PredictionView from './views/PredictView';
import DatasetView from './views/DatasetView';
import TrainView from './views/TrainView';


const App: Component = () => {
  return (
    <div class={styles.App}>
      <TopBar />


      <section id='layout'>
      <PanelsButton />
      <Switch>
        <Match when={onPanel() == Panels.prediction}>
          <PredictionView />
        </Match>

        <Match when={onPanel() == Panels.data}>
          <DatasetView />
        </Match>

        <Match when={onPanel() == Panels.train}>
          <TrainView />
        </Match>
      </Switch>
      </section>
    </div>
  );
};

export default App;

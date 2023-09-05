import type { Component } from 'solid-js';

import styles from './App.module.css';
import TopBar from './components/top-bar/TopBar';
import FileSelection from './components/file-selection/FileSelection';

const App: Component = () => {
  return (
    <div class={styles.App}>
      <TopBar />

      <section id='layout'>
      <FileSelection onSelect={(e) => {
        console.log("test", e);
        
      }}/>
      </section>

    </div>
  );
};

export default App;

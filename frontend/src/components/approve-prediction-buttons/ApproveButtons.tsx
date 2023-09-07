import { createSignal } from 'solid-js';
import "./ApproveButtons.css"

export default function () {
  const GENRE_CHOICES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'];

  return (
    <div id="audi-card" class="w-[250px]">
      {GENRE_CHOICES.map((genre, index) => (
        <button>{genre}</button>
      ))}
    </div>
  );
}

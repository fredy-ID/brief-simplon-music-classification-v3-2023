import Api from "../services/api.service"
import { createSignal } from 'solid-js';

export default function(){
    const [trainingMessage, setTrainingMessage] = createSignal('');
    const [trainingEpochs, setTrainingEpochs] = createSignal('');
    const [trainingAccuracy, setTrainingAccuracy] = createSignal('');
    const [trainingDataCount, setTrainingDataCount] = createSignal('');
    const [trainingGenres, setTrainingGenres] = createSignal('');
    const [genreCounts, setGenreCounts] = createSignal<{ [key: string]: number }>({});
    const [isTraining, setIsTraining] = createSignal(false);

    const onClickTrain = async () => {
        setIsTraining(true);
        try {
            const response = await Api.post('/train-model/', {}, true);
            console.log(response);
            setTrainingMessage(response.msg);
            setTrainingEpochs(response.epochs);
            setTrainingAccuracy(response.accuracy);
            setTrainingDataCount(response.num_train_samples);
            setTrainingGenres(response.genres_used_for_training.join(', '));
            setGenreCounts(response.genre_counts);
        } catch (error) {
            console.error('Erreur lors de la requête API :', error);
            setTrainingMessage('Erreur lors de l\'entraînement du modèle.'); // Affichez un message d'erreur en cas d'échec de la requête
        }
        setIsTraining(false);
    };

    return <section>
        <button onClick={onClickTrain}>Entrainer</button>
        {isTraining() && (
            <p>Entraînement en cours...</p>
        )}
        
        {trainingMessage() && (
            <div>
                <p>{trainingMessage()}</p>
                <p>Données d'entraînement : {trainingDataCount()} musiques ({trainingGenres()})</p>
                <p>Entraîné sur {trainingEpochs()} epochs</p>
                <hr />
                <p>L'accuracy est de {trainingAccuracy()} %</p>
            </div>
        )}

        {Object.entries(genreCounts()).map(([genre, count], index) => (
            <p key={index}>{genre}: {count}</p>
        ))}
    </section>
}
import Api from "../services/api.service"
import "./TrainView.css"
import { createSignal } from 'solid-js';

export default function(){
    const [trainingMessage, setTrainingMessage] = createSignal('');
    const [trainingEpochs, setTrainingEpochs] = createSignal('');
    const [trainingAccuracy, setTrainingAccuracy] = createSignal('');
    const [trainingDataCount, setTrainingDataCount] = createSignal('');
    const [testDataCount, setTestDataCount] = createSignal('');
    const [trainingGenres, setTrainingGenres] = createSignal('');
    const [genreCounts, setGenreCounts] = createSignal<{ [key: string]: number }>({});
    const [isTraining, setIsTraining] = createSignal(false);
    const [limit, setLimit] = createSignal(1); // Ajoutez le state pour la limite
    

    const onClickTrain = async () => {
        setIsTraining(true);
        try {
            const response = await Api.post('/train-model/', { limit: limit() }, true);
            console.log(response);
            setTrainingMessage(response.msg);
            setTrainingEpochs(response.epochs);
            setTrainingAccuracy(response.accuracy);
            setTrainingDataCount(response.num_train_samples);
            setTestDataCount(response.num_test_samples);
            setTrainingGenres(response.genres_used_for_training.join(', '));
            setGenreCounts(response.genre_counts);
        } catch (error) {
            console.error('Erreur lors de la requête API :', error);
            setTrainingMessage('Erreur lors de l\'entraînement du modèle.'); // Affichez un message d'erreur en cas d'échec de la requête
        }
        setIsTraining(false);
    };

    const handleLimitChange = (e: Event) => {
        const target = e.target as HTMLInputElement;
        setLimit(parseInt(target.value, 10));
        console.log(limit())
    };

    return <section>
        {!isTraining() && (
            <div>
                <button onClick={onClickTrain}>Entrainer</button>
                <div>
                    <label class="limit">Limit :</label>
                    <input type="number" id="limit" value={limit()} onChange={handleLimitChange} />
                </div>
            </div>
            
        )}
        {isTraining() && (
            <p class="text-3xl font-bold">Entraînement en cours...</p>
        )}
        {!isTraining() && trainingMessage() && (
            <div>
                <p class="text-3xl font-bold my-5">{trainingMessage()}</p>
                <p>Données d'entraînement : <span class="font-bold">{trainingDataCount()} musique(s)</span> ({trainingGenres()})</p>
                <p>Données de test : <span class="font-bold">{testDataCount()} musique(s)</span></p>
                <p>Entraîné sur <span class="font-bold">{trainingEpochs()} epochs</span></p>
                <hr class="my-3" />
                <p>L'accuracy est de {trainingAccuracy()} %</p>
            </div>
        )}

        {!isTraining() && Object.entries(genreCounts()).map(([genre, count], index) => (
            <p class="font-bold" key={index}>{genre}: {count}</p>
        ))}

        
    </section>
}
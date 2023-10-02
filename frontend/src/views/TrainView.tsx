import Api from "../services/api.service"
import { createSignal } from 'solid-js';

export default function(){
    const [trainingMessage, setTrainingMessage] = createSignal('');
    const [trainingEpochs, setTrainingEpochs] = createSignal('');
    const [trainingAccuracy, setTrainingAccuracy] = createSignal('');
    const [isTraining, setIsTraining] = createSignal(false);

    const onClickTrain = async () => {
        setIsTraining(true);
        try {
            const response = await Api.post('/train-model/', {}, true);
            console.log(response);
            setTrainingMessage(response.msg); // Mettez à jour le message de formation
            setTrainingEpochs(response.epochs); // Mettez à jour le message de formation
            setTrainingAccuracy(response.accuracy); // Mettez à jour le message de formation
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
                <p>Données d'entraînement : __ </p>
                <p>Entraîné sur {trainingEpochs()} epochs</p>
                <p>L'accuracy est de {trainingAccuracy()} %</p>
            </div>
        )}
    </section>
}
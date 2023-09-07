import Api from "../services/api.service"

export default function(){
    const onClickTrain = async () => {
        const json = Api.post("/train-model/", {})
        console.log(json);
        
    }

    return <section>
        <button onClick={onClickTrain}>Entrainer</button>
    </section>
}
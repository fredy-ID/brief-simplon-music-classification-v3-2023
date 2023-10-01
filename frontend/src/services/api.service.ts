const config = {
    // host: "http://127.0.0.1:8000",
    host: "http://simplon.fredy-mc.fr/backend-music-classification-v3",
    baseUrl: "/api",
    options: {
        method: "GET",
        
    }
}

export default class Api {

    static async generic(url: string, options: object, haveContentType = false){
        const URL = config.host + config.baseUrl + url;
        let response: any = undefined;
        let contentType = {
            
        }
        
        if(haveContentType){
            contentType = {
                headers: {
                    "Content-Type": "application/json",
                },
            }
        }

        try{
            response = await fetch(URL, {...config.options, ...contentType, ...options})
        }catch(error){
            console.log(error);
            return false
        }

        if(response != undefined){            
            try{
                const json = await response.json()
                return json
            }catch(error){
                console.log("error", error);
                return false
            }
        }
    }

    static async get(url: string){
        const json = await Api.generic(url, {
            method: "GET"
        })

        if(!json) return false
        return json
    }


    static async post(url: string, body: object, haveContentType = true){
        let data;

        if(body instanceof FormData){
            data = body
        }else{
            data = JSON.stringify(body)
        }
        
        const json = await Api.generic(url, {
            method: "POST",
            body: data,
        }, haveContentType)
        
        if(!json) return false
        return json
    }
}
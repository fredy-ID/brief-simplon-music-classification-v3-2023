const config = {
    host: "http://simplon.fredy-mc.fr/backend-music-classification-v3/api",
    options: {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }
}

export default class Api {

    static async generic(url: string, options: object){
        let response: any = undefined;
        try{
            response = await fetch(config.host + url, {...config.options, ...options})
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


    static async post(url: string, body: object){
        let data;

        if(body instanceof FormData){
            data = body
        }else{
            data = JSON.stringify(body)
        }
        
        const json = await Api.generic(url, {
            method: "POST",
            body: data
        })
        
        if(!json) return false
        return json
    }
}
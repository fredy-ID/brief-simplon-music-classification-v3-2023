import { createEffect } from "solid-js";
import { file } from "../file-selection/FileSelection";
import "./AudioCard.css"

export default function (){
    const audioImage = "https://imgs.search.brave.com/0yVsGV7Tx3dCXrZ9lgFdpdOYGSUYOJrDuQYVgEt49_s/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9pbWcu/ZnJlZXBpay5jb20v/ZnJlZS12ZWN0b3Iv/M2QtYXVkaW8tc291/bmR3YXZlXzEyMTct/MzE3Ni5qcGc_c2l6/ZT02MjYmZXh0PWpw/Zw"
    
    createEffect(() => {
        if(file() != undefined){
            console.log(file());   
        }
    })

    return <div id="audi-card" class="w-[250px]">
        <div class="img-container">
            <img class="img" src={audioImage} alt="" />
        </div>
        <div class="w-full overflow-hidden">
            
        </div>
    </div>
}
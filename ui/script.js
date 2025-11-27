const video = document.getElementById("video");
const letraLabel = document.getElementById("letra");
const fraseLabel = document.getElementById("frase");

let letraAtual = "?";
let fraseAtual = "";

// Captura da webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Erro ao acessar webcam: ", err);
    });

// Simulação do reconhecimento de letra (substituir depois pelo TF.js)
function detectarLetra() {
    // Aqui você integraria seu modelo TensorFlow.js
    const letras = ["A","B","C","D","E","F","G","H","I","J"];
    letraAtual = letras[Math.floor(Math.random() * letras.length)];
    letraLabel.textContent = `Letra atual: ${letraAtual}`;
}

setInterval(detectarLetra, 1000); // atualiza letra a cada 1s

// Eventos de teclado
document.addEventListener("keydown", (e) => {
    if(e.key === "Enter") {
        if(letraAtual !== "?") fraseAtual += letraAtual;
    } else if(e.key === " ") {
        fraseAtual += " ";
    } else if(e.key === "Backspace") {
        fraseAtual = fraseAtual.slice(0, -1);
    } else if(e.key === "Control") {
        if(fraseAtual) {
            const utterance = new SpeechSynthesisUtterance(fraseAtual);
            speechSynthesis.speak(utterance);
        }
    } else if(e.key === "Escape") {
        fraseAtual = "";
    }

    fraseLabel.textContent = `Meu nome é ${fraseAtual || "(?)"}`;
});

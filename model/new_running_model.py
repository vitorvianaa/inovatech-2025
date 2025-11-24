import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
from keras.models import load_model
import pickle
from audio.audio import generate_audio, play_audio
import keyboard


ctk.set_appearance_mode("dark")

AZUL_FUNDO     = "#0a0f1a"
AZUL_CONTAINER = "#111827"
AZUL_CARD      = "#1a2335"
DOURADO        = "#d4af37"
DOURADO_SUAVE  = "#b9972d"
BRANCO_SUAVE   = "#e6e6e6"

model = load_model('best_model.h5')

with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

janela = ctk.CTk()
janela.geometry("820x820")
janela.resizable(False, False)
janela.title("Silent Voice")
janela.configure(fg_color=AZUL_FUNDO)


container = ctk.CTkFrame(
    janela, width=780, height=780,
    fg_color=AZUL_CONTAINER, corner_radius=30
)
container.pack(pady=20)
container.pack_propagate(False)

ctk.CTkLabel(
    container,
    text="Silent Voice",
    font=ctk.CTkFont(size=40, weight="bold"),
    text_color=DOURADO
).pack(pady=(20,10))

ctk.CTkLabel(
    container,
    text="Demonstração em tempo real, ENGCO222M01",
    font=ctk.CTkFont(size=18),
    text_color=DOURADO_SUAVE
).pack()


frame_camera = ctk.CTkFrame(
    container, width=740, height=480,
    corner_radius=25, fg_color=AZUL_CARD
)
frame_camera.pack(pady=25)
frame_camera.pack_propagate(False)

camera_label = ctk.CTkLabel(frame_camera, text="")
camera_label.pack(expand=True)

frame_resultado = ctk.CTkFrame(
    container, width=740, height=180,
    corner_radius=25, fg_color=AZUL_CARD
)
frame_resultado.pack(pady=10)
frame_resultado.pack_propagate(False)

resultado_label = ctk.CTkLabel(
    frame_resultado,
    text="Meu nome é (?)",
    font=ctk.CTkFont(size=46, weight="bold"),
    text_color=DOURADO
)
resultado_label.pack(pady=10)

letra_label = ctk.CTkLabel(
    frame_resultado,
    text="Letra atual: (?)",
    font=ctk.CTkFont(size=28),
    text_color=BRANCO_SUAVE
)
letra_label.pack()

frase_atual = ""
letra_atual = "?"
confianca_atual = 0.0

cap = cv2.VideoCapture(0)

def atualizar_camera():
    global letra_atual, frase_atual, confianca_atual

    ret, frame = cap.read()
    if not ret:
        camera_label.after(10, atualizar_camera)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)


    if results.multi_hand_landmarks:
        for mao in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, mao, mp_hands.HAND_CONNECTIONS)

            pontos = []
            for lm in mao.landmark:
                pontos.extend([lm.x, lm.y, lm.z])

            entrada = np.array([pontos])
            pred = model.predict(entrada, verbose=0)

            idx = np.argmax(pred)
            confianca_atual = float(np.max(pred))

            letra_atual = encoder.inverse_transform([idx])[0] if confianca_atual > 0.7 else "?"

    else:
        letra_atual = "?"
        confianca_atual = 0.0


    if keyboard.is_pressed("enter"):
        if letra_atual != "?":
            frase_atual += letra_atual

    if keyboard.is_pressed("space"):
        frase_atual += " "

    if keyboard.is_pressed("backspace"):
        frase_atual = frase_atual[:-1]

    if keyboard.is_pressed("esc"):
        janela.destroy()

    if keyboard.is_pressed("ctrl"):
        if frase_atual:
            file_path = generate_audio(name=frase_atual)
            play_audio(file_path)


    resultado_label.configure(text=f"Meu nome é {frase_atual if frase_atual else '(?)'}")
    letra_label.configure(text=f"Letra atual: {letra_atual}")


    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((740, 480))
    imgtk = ImageTk.PhotoImage(img)

    camera_label.img = imgtk
    camera_label.configure(image=imgtk)

    camera_label.after(10, atualizar_camera)


def mostrar_tela_comandos():
    popup = ctk.CTkToplevel(janela)
    popup.title("Comandos disponíveis")
    popup.geometry("500x400")
    popup.resizable(False, False)
    popup.configure(fg_color=AZUL_CONTAINER)
    popup.grab_set()

    ctk.CTkLabel(
        popup,
        text="Comandos do Tradutor",
        font=ctk.CTkFont(size=28, weight="bold"),
        text_color=DOURADO
    ).pack(pady=(20,10))

    comandos = (
        "ENTER  → Adiciona a letra detectada\n"
        "ESPAÇO → Adiciona um espaço\n"
        "BACKSPACE → Apaga a última letra\n"
        "CTRL → Gerar e tocar áudio\n"
        "ESC → Fechar o programa"
    )

    ctk.CTkLabel(
        popup,
        text=comandos,
        font=ctk.CTkFont(size=20),
        text_color=BRANCO_SUAVE,
        justify="left"
    ).pack(pady=20)

    def iniciar():
        popup.destroy()
        atualizar_camera()

    ctk.CTkButton(
        popup,
        text="Começar",
        fg_color=DOURADO,
        hover_color=DOURADO_SUAVE,
        text_color="black",
        font=ctk.CTkFont(size=20, weight="bold"),
        command=iniciar
    ).pack(pady=20)


mostrar_tela_comandos()
janela.mainloop()

cap.release()
cv2.destroyAllWindows()

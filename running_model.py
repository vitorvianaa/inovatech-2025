import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import pickle

# --- CONFIGURAÇÕES INICIAIS ---

# 1. Carregar o Modelo Keras (.h5)
print("Carregando modelo neural...")
model = load_model('best_model.h5')

# 2. Carregar o Label Encoder (para traduzir Numero -> Letra)
try:
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    print("Modelos carregados com sucesso!")
except FileNotFoundError:
    print("ERRO: 'label_encoder.pkl' não encontrado.")
    print("Você precisa ter salvo o encoder durante o treinamento.")
    exit()

# Configurações do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

frase_atual = ""
letra_atual = "?"
confianca_atual = 0.0

cap = cv2.VideoCapture(0)

print("--- CONTROLES ---")
print("[ENTER] -> Adiciona letra")
print("[ESPAÇO] -> Adiciona espaço")
print("[BACKSPACE] -> Apaga")
print("[ESC] -> Sair")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lista_pontos = []
            for lm in hand_landmarks.landmark:
                lista_pontos.extend([lm.x, lm.y, lm.z])
            
            entrada_modelo = np.array([lista_pontos])
            
            #  probabilidades para todas as letras
            predicao = model.predict(entrada_modelo, verbose=0)
            
            # índice da maior probabilidade
            indice_maximo = np.argmax(predicao)
            confianca_atual = np.max(predicao)
            
            # > 70% de certeza
            if confianca_atual > 0.7:
                letra_atual = encoder.inverse_transform([indice_maximo])[0]
            else:
                letra_atual = "?"

            # Exibir na tela
            cv2.putText(frame, f"Sinal: {letra_atual} ({confianca_atual*100:.1f}%)", 
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        letra_atual = "?"
        confianca_atual = 0.0

    # interface
    cv2.rectangle(frame, (0,0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Frase: {frase_atual}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == 13: # ENTER
        if letra_atual != "?":
            frase_atual += letra_atual
    elif key == 32: # ESPAÇO
        frase_atual += " "
    elif key == 8: # BACKSPACE
        frase_atual = frase_atual[:-1]
    elif key == 27: # ESC
        break

    cv2.imshow('Tradutor LIBRAS - Deep Learning', frame)

cap.release()
cv2.destroyAllWindows()
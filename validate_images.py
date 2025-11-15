import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model


model = load_model('modelo_maos.h5')
labels = ['A', 'B', 'C', 'D', 'NONE']
IMG_SIZE = 128
threshold = 0.7  # confiança mínima


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Pressione ENTER para capturar e classificar, ESC para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    
    cv2.imshow("Classificacao com Imagens", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  
        break
    elif key == 13:  # ENTER -> capturar e classificar
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # get bounding box 
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

                
                margin = 20
                x_min = max(x_min - margin, 0)
                y_min = max(y_min - margin, 0)
                x_max = min(x_max + margin, w)
                y_max = min(y_max + margin, h)

              
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    print("ROI vazia — mão fora do quadro.")
                    continue

                img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)

                
                prediction = model.predict(img)[0]

                # Pega o top 3
                top_indices = np.argsort(prediction)[::-1][:3]
                top_values = prediction[top_indices]
                top_labels = [labels[i] for i in top_indices]
                
                # Mostra a diferença entre as mais prováveis
                print("Top 3 predições:")
                for lbl, val in zip(top_labels, top_values):
                    print(f"{lbl}: {val*100:.2f}%")

                # filtra pelas classes acima do threshold
                indices = np.where(prediction >= threshold)[0]
                indices = indices[np.argsort(prediction[indices])[::-1]]  # ordena do maior p menor
                top_filtered = [(labels[i], float(prediction[i])) for i in indices]

                print("Prediction:", prediction)
                print(f'top filtered: {top_filtered}')
                if len(top_filtered) == 0:
                    print("Sem classificação acima do threshold.")
                    print(f'Voce quis dizer: {top_labels}?')
                elif len(top_filtered) == 1:
                    label, conf = top_filtered[0]
                    if label == 'NONE':
                        print(f'Sem classificacao')
                    else:
                        print(f"Classificação: {label} ({conf*100:.2f}%)")
                else:
                    sugestoes = " ou ".join([f"{lbl} ({conf*100:.1f}%)" for lbl, conf in top_filtered])
                    print(f"Você quis dizer: {sugestoes}?")

                
                cv2.imshow("Regiao da Mao", roi)

cap.release()
cv2.destroyAllWindows()

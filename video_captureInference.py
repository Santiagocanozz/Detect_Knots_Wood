import cv2
import json
import numpy as np
import os

def process_frame(frame):
    JSON_config = open('C:/Users/Usuario/Documents/Proyectos/Detect_Knots_Wood/propertiesYOLO.json')
    dataLoad = json.load(JSON_config)

    weights = dataLoad['data']['weights']
    cfg = dataLoad['data']['cfg']
    names = dataLoad['data']['obj_names']

    # Cargamos el modelo YOLO pre-entrenado y sus pesos
    net = cv2.dnn.readNet(weights, cfg)

    # Cargamos las clases a las que YOLO puede detectar
    classes = []
    with open(names, 'r') as f:
        classes = f.read().strip().split('\n')

    # Escalamos y preprocesamos el cuadro del video
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Definimos las capas de salida
    layer_names = net.getUnconnectedOutLayersNames()

    # Realizamos la inferencia
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Lista para almacenar información de detección
    class_ids = []
    confidences = []
    boxes = []

    # Procesamos las detecciones
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Umbral de confianza
                # Coordenadas del cuadro delimitador
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Puntos de referencia del cuadro delimitador
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Aplicamos la supresión de no máximos para eliminar detecciones superpuestas
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    # Dibujamos los cuadros delimitadores en el cuadro del video
    for i in indices:
        i = i[0]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        box = boxes[i]
        color = (0, 255, 0)  # Color del cuadro delimitador en formato BGR
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Ruta del video de entrada
video_path = "C:/Users/Usuario/Downloads/test.mp4"

# Abre el video para lectura
cap = cv2.VideoCapture(video_path)

# Verifica si el video se abrió correctamente
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

# Crea un objeto VideoWriter para guardar el video procesado
acceleration_factor = 100

# Obtiene la tasa de fotogramas original
original_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Calcula la nueva tasa de fotogramas
new_fps = original_fps * acceleration_factor

# Crea un objeto VideoWriter para guardar el video acelerado
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
new_width = 640  # Cambia esto al ancho deseado
new_height = 480 
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (new_width, new_height))
    if not ret:
        break

    processed_frame = process_frame(frame)
    #out.write(processed_frame)

    cv2.imshow("YOLO Object Detection", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

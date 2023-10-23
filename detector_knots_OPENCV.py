import cv2
import json
import numpy as np

JSON_config = open('propertiesYOLO.json')
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

# Cargamos la imagen que deseas analizar
image = cv2.imread('C:/Users/Usuario/Documents/Proyectos/Detect_Knots_Wood/Detect_Knots_Wood/examples_images/FLIR1059.jpg')

# Escalamos y preprocesamos la imagen
blob = cv2.dnn.blobFromImage(image, 1/255, (832, 832), swapRB=True, crop=False)

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
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])

            # Puntos de referencia del cuadro delimitador
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Aplicamos la supresión de no máximos para eliminar detecciones superpuestas
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

# Dibujamos los cuadros delimitadores en la imagen
for i in indices:
    i = i[0]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    box = boxes[i]
    color = (0, 255, 0)  # Color del cuadro delimitador en formato BGR
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
    cv2.putText(image, f'{label} {confidence:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Mostramos la imagen con las detecciones
cv2.imshow('YOLO Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import os

#carpeta almacena imagenes
nombre = 'imagenes_detector'
direccion = 'C:/CSC-2025'
carpeta = os.path.join(direccion, nombre)

if not os.path.exists(carpeta):
    print("Carpeta creada")
    os.makedirs(carpeta)

#inicio contador
contador = 0

#declaramos detector
detector = mp.solutions.face_detection
dibujo = mp.solutions.drawing_utils

#videocaptura
cap = cv2.VideoCapture(1)

#Inicia parametros de deteccion
with detector.FaceDetection(min_detection_confidence= 0.75) as rostros: #min_detection_confidence umbral minimo detección
    while True:
        # ret :lectura fotogramas correcta,frame todos los fotogramas
        ret, frame = cap.read()#lectura de videocaptura

        #guia movimiento para servomotor
        frame = cv2.flip(frame, 1)

        #correcion color BGR a RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #deteccion de rostros
        resultado = rostros.process(rgb) #rgb son los frames cambiados

        #filtro de seguridad
        if resultado.detections is not None:
            for rostro in resultado.detections: #lee cada rostro que encuentra
                #dibujo.draw_detection(frame, rostro, dibujo.DrawingSpec(color=(0,255,0)))#dibuja un cuadro en rostro, muestra en frame

                for id, coordenadas in enumerate(resultado.detections):

                    #dimensiones de imagen
                    al, an, c = frame.shape #medidas de fotogramas

                    #Extraer x inicial
                    x = coordenadas.location_data.relative_bounding_box.xmin
                    y = coordenadas.location_data.relative_bounding_box.ymin

                    #extraer ancho y alto
                    ancho = coordenadas.location_data.relative_bounding_box.width
                    alto = coordenadas.location_data.relative_bounding_box.height

                    #conversion a pixeles
                    xi = int(x * an)
                    yi = int(y * al)
                    ancho = int(ancho * an)
                    alto = int(alto * al)

                    #hallamos Xfinal y YFinal
                    xf = xi + ancho
                    yf = yi + alto

                    #extraccion de pixeles
                    cara = frame[yi:yf, xi:xf]


                    #extraer el punto central del rostro
                    c_central_x = (xi + (xi + xf)) // 2
                    c_central_y = (yi + (yi + yf)) // 2

                    #redimensionar las imagenes
                    cara = cv2.resize(cara, (150,200), interpolation=cv2.INTER_CUBIC)

                    #almacena imagenes de rostros
                    cv2.imwrite(carpeta + "/rostro_{}.jpg".format(contador), cara)
                    contador += 1
                    #Mostrar coordenadas
                    cv2.circle(frame, (c_central_x, c_central_x), 5, (255,0,255), cv2.FILLED)

        #mostramos el fotograma, detecta a 1 metro de camara
        cv2.imshow("Reconocimiento Facial", cara)

        #Leemos una tecla ,espera 1 para leer una orden
        t = cv2.waitKey(1)

        if t == 27 or contador > 30: #ASCI 27 tecla ESC, cambiar a mas de 30
            break
cap.release() #borra las videocapturas
cv2.destroyAllWindows()#cierra
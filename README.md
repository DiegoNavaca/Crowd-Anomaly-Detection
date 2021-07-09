# Crowd-Anomaly-Detection
Modelo de aprendizaje automático para detectar anomalías en vídeos de multitudes. Ha sido probado para la detección de peleas y escenas de pánico.
En la memoria adjunta puede encontrar más información sobre el funcionamiento y los resultados obtenidos. 
También se dispone de una página web en la que se muestran los resultados, y que es accesible a través del siguiente enlace: https://diegonavaca.github.io/Crowd-Anomaly-Detection.

Los archivos se dividen de la siguiente manera:
### descriptors.py
Incluye todo lo relacionado con la extracción de des-
criptores de un vídeo. Una vez procesado, sus descriptores serán guar-
dados en un archivo para facilitar su tratamiento y limitar el uso de
memoria.
### visualization.py
Continene funciones que permiten visualizar distintos aspectos del procesamiento del vídeo, como el grafo de Delaunay o
las trayectorias.
### files.py
Incluye funciones para la lectura del ground truth de cada con-
junto de datos, así como de las etiquetas de cada vídeo. También incluye
una función para extraer los descriptores de un conjunto de vídeos y la
preparar las etiquetas de estos de manera acorde.
### data.py
Incluye funciones que leen la información guardada en los ar-
chivos y devuelven el vector de histogramas y etiquetas que se usarán
para entrenar el modelo.
### models.py
Contiene todo lo referente al entrenamiento y evaluación de
los modelos de codificación y clasificación, a partir de los histogramas
y etiquetas ya preparados.
### autoencoders.py
Contiene la definición del autoencoder usado para co-
dificar los histogramas, con parámetros para ajustar diversos aspectos
de su diseño, como el número de capas o la función de activación de las
capas ocultas.
### main.py
Centro de la implementación. Usa las funciones del resto de
archivos con los parámetros adecuados para cada conjunto de datos.
### optimize_params.py
Contiene la información relacionada con el uso
de Optuna para el diseño del autoencoder.

## Cómo usar un modelo entrenado.
Para aplicar un modelo entrenado a un vídeo se debe incluir un diccionario en "others" con la clave "model". Este modelo será cargado al inicio de la función "extract_descriptors" y se usará para colocar un borde rojo alrededor de los fotogramas anómalos del vídeo que se esté procesando.
El diccionario deberá contener los siguientes parámetros: 
#### "n_bins"
El número de divisiones de los histogramas.
#### "ranges"
El nombre del archivo en el que esten guardados los rangos de los histogramas, primero los valores máximos de todos ellos y luego los mínimos.
#### "classiffier"
El nombre del archivo en el que se guarda el SVM ya entrenado.
#### "codifier"
Opcionalmente se puede añadir el nombre del archivo en el que se guarda el autoencoder ya entrenado.

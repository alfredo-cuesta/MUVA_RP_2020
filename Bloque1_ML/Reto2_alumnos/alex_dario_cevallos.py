#!/usr/bin/env python
# coding: utf-8

# # Reto 2: Problema multiclase
# 
# Este reto consiste en aprender a clasificar 4 tipos diferentes de vehículos utilizando cualquiera de los clasificadores o técnicas estudiadas hasta el momento. Esto incluye:
# + clasificación lineal
# + transformaciones no lineales seguido de un clasificador lineal
# + Support Vector Machines (SVM)
# + Decision Tree (DT)
# 
# Además se pueden aplicar técnicas de preprocesado como:
# + escalado de las características
# + *grid search* para búsqueda de hiperparámetros
# + validación cruzada
# 
# El conjunto de datos, *vehiculos_reto2.csv*, consiste en 592 muestras de vehículos; cada uno de ellos representado por 18 características.
# 
# Para evaluar las propuestas se utilizará un conjunto de datos que se mantendrá oculto hasta después de la entrega
# 
# ### Requisitos
# + La entrega se realiza **sólo** a través de la tarea habilitada para ello en la pestaña de *Evaluación* del Aula Virtual.
# + Se debe entregar un cuaderno Jupyter con el nombre de los participantes.<br>
#   *Por ejemplo*:   **Cuesta_LeCunn.ipynb**
# + El cuaderno entregado debe seguir la estructura y reglas de este cuaderno
# 
# ### Competición
# + Todos los cuadernos entregados se subirán al repo de GitHub y se ejecutarán en Binder, donde ya estará en conjunto de test que permanecía oculto.
# + El número de aciertos respecto del número de ejemplos será la puntuación del reto.
# + **Importante** Es muy fácil asegurarte de que tu código funcionará bien. Para ello:
#     1. Agrupa todo tu código en una única celda
#     2. En el cuaderno del reto que hay en Binder: elimina las celdas que hay entre la verde y la roja, y copia tu celda entre ellas.
#     3. Ejecuta ese cuaderno de Binder. 
#     
# ### Plazo: lunes 26 de oct. de 2020 a las 6 am.
# Es decir, incluye toda la noche del domingo 25 de oct.
# 
# 
# ---
#     [ES] Código de Alfredo Cuesta Infante para 'Reconocimiento de Patrones'
#        @ Master Universitario en Visión Artificial, 2020, URJC (España)
#     [EN] Code by Alfredo Cuesta-Infante for 'Pattern Recognition'
#        @ Master of Computer Vision, 2020, URJC (Spain)
# 
#     alfredo.cuesta@urjc.es

# In[1]:


# Conjunto distribuido para el reto

Challange_filename = '../../Datasets/vehiculos_reto2.csv'


# In[2]:


# Conjunto NO distribuido para evaluar los clasificadores entregados

Test_filename = '../../Datasets/vehiculos_reto2.csv' #<-- este nombre cambiará después del plazo de entrega


# In[3]:


#-[1]. Load data from CSV and put all in a single dataframe 'FullSet'

import numpy  as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.append('../../MyUtils/')
import MyUtils as my
seed = 1234 #<- random generator seed (comment to get randomness)

#-[2]. Load data from CSV and put all in a single dataframe 'FullSet'

FullSet = pd.read_csv(Challange_filename, header=0)
FullX = FullSet.drop('Class', axis=1)
FullY = FullSet[['Class']]


# <table style="width:100%;"> 
#  <tr style='background:lime'>
#   <td style="text-align:left">
#       <h2>Tu código debe empezar a partir de aquí y puede tener tantas celdas como quieras</h2>
#       <p> Si quieres, puedes borrar (o convertir en RawNBConvert) las celdas de ejemplo
#       <h3>Importante:</h3>
#       <p>Tu código debe producir las siguientes variables: </p>
#       <p> $\quad \bullet$ <b>clf:</b> el clasificador final con el que se realizará el test<br>
#        $\quad \bullet$ <b>X_test:</b> el conjunto de test listo para ser usado por el método <b>predict</b><br>
#        $\quad \bullet$ <b>Y_test:</b> es el vector de etiquetas del conjunto de X_test listo para ser usado por el método <b>confusion_matrix</b>
#       </p>
#   </td>
#  </tr>
# </table>

# In[4]:


nombres = ["Alex Cevallos"]


# In[5]:


def feat_extraction (data, theta=0.1):
    # data: dataframe
    # theta: parameter of the feature extraction
    # features extracted: 
    #   'width','W_max1','W_max2','W_max3',
    #   'height','H_max1','H_max2','H_max3',
    #   'area','w_vs_h'
    #
    features = np.zeros([data.shape[0], 10]) #<- allocate memory with zeros
    data = data.values.reshape([data.shape[0],28,28]) 
    #-> axis 0: id of instance, axis 1: width(cols) , axis 2: height(rows)
    for k in range(data.shape[0]):
        #..current image 
        x = data[k,:,:]
        #--width feature
        sum_cols = x.sum(axis=0) #<- axis0 of x, not of data!!
        indc = np.argwhere(sum_cols > theta * sum_cols.max())
        col_3maxs = np.argsort(sum_cols)[-3:] 
        features[k,0] = indc[-1] - indc[0]
        features[k,1:4] = col_3maxs
        #--height feature
        sum_rows = x.sum(axis=1) #<- axis1 of x, not of data!!
        indr = np.argwhere(sum_rows > theta * sum_rows.max())
        features[k,4] = indr[-1] - indr[0]
        row_3maxs = np.argsort(sum_rows)[-3:] 
        features[k,5:8] = row_3maxs
      #--area
    features[:,8] = features[:,0] * features[:,4]
    #--ratio W/H
    features[:,9] = features[:,0] / features[:,4]
    col_names = ['width','W_max1','W_max2','W_max3','height','H_max1','H_max2','H_max3','area','w_vs_h']
    #
    return pd.DataFrame(features,columns = col_names) 


# In[6]:


#-- ejemplo de preprocesado --
feat_1 = 'width'
feat_2 = 'height'
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(FullX)
Y = FullY.values.ravel() 


# In[7]:


#-- ejemplo de entrenamiento --

from sklearn.svm import SVC

clf = SVC(kernel='rbf', C=0.01, gamma=0.01, random_state = seed)
clf.fit( X, Y )

'''RESULTADO: clf es el objeto con el clasificador'''


# In[8]:


#-- ejemplo de test --
from sklearn.model_selection import StratifiedShuffleSplit

test_size = 0.01
splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
split_ix = splitter.split(FullX,FullY)
for train_ix, test_ix in split_ix:
    X_train = FullX.loc[train_ix].reset_index(drop=True)
    Y_train = FullY.loc[train_ix].reset_index(drop=True)
    X_test  = FullX.loc[test_ix].reset_index(drop=True)
    Y_test  = FullY.loc[test_ix].reset_index(drop=True)
#-la evaluación se realiza en las celdas de abajo

'''RESULTADO: X_test es el dataframe para utilizar en >>> Y_pred = clf.predict() 
   RESULTADO: Y_test es el array con las etiquetas para utilizar en >>> confusion_matrix(Y_test,Y_pred)
'''


# <table style="width:100%;"> 
#  <tr style='background:pink'>
#   <td style="text-align:left">
#       <h2>A partir de aquí ya no se pueden modificar las celdas</h2>
#           <h3>Comprueba que:</h3>
#           <p> $\quad \bullet$ tu clasificador está almacenado en la variable <b>clf</b><br>
#               $\quad \bullet$ tienes el conjunto de test correctamente almacenado en la variable <b>X_test</b><br>
#               $\quad \bullet$ tienes las etiquetas del conjunto de test correctamente almacenadas en la variable <b>Y_test</b><br>
#           </p>
#       
#   </td>
#  </tr>
# </table>

# ## Test

# In[9]:


from sklearn.metrics import confusion_matrix

Y_hat = clf.predict(X_test)
conf_mat = confusion_matrix(Y_test , Y_hat)
N_success  = np.trace(conf_mat)
N_fails = Y_test.shape[0]-N_success
#-------------------------------
print (nombres,"\n")
print("Confusion matrix:\n")
print(conf_mat,"\n")
print("Outcome:\n")
strlog = "  :) HIT  = %d, (%0.2f%%)"%(N_success, 100*N_success/(N_success+N_fails))
print(strlog)
strlog = "  :( FAIL = %d, (%0.2f%%)"%(N_fails, 100*N_fails/(N_success+N_fails))
print(strlog)


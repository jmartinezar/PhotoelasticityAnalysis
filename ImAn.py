'''
Import all libraries
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import os
import pandas as pd

'''

'''
def tw(x, s, max, min):
  return np.arctan(max*np.sin(x - s)/(min*np.cos(x - s)))

'''

'''
def ellipse(x, s, max, min):
  return np.sqrt( (max*np.cos(tw(x, s, max, min)))**2  + (min*np.sin(tw(x, s, max, min)))**2 )

'''
Define function to read image to folder
'''
def read_to_folder(directory, x, y, m):
  results = []
  # Leer archivos en la carpeta
  for File in os.listdir(directory):
      if File.startswith('Fotoelasticidad_') and File.endswith('.jpeg'):
          # Obtener el número del nombre del archivo
          number = int(File.split('_')[1].split('.')[0])

          # Construir la ruta completa del archivo
          file_path = os.path.join(directory, File)

          # Leer la imagen en escala de grises
          image = cv2.imread(file_path, cv2.IMREAD_COLOR)

          blue_channel, green_channel, red_channel = cv2.split(image)

          # Calcular el promedio de los pixeles en la ventana centrada en (x,y)
          window_b = blue_channel[y - m // 2:y + m // 2 + 1, x - m // 2:x + m // 2 + 1]
          mean_b = np.mean(window_b)

          window_g = green_channel[y - m // 2:y + m // 2 + 1, x - m // 2:x + m // 2 + 1]
          mean_g = np.mean(window_g)

          window_r = red_channel[y - m // 2:y + m // 2 + 1, x - m // 2:x + m // 2 + 1]
          mean_r = np.mean(window_r)

          # Guardar el número y el promedio en la lista de resultados
          results.append([number, mean_b, mean_g, mean_r])

  # Convertir la lista de resultados a un arreglo de Numpy
  results = np.array(results)
  return results

'''

'''

def obtain_data(directory, x, y, m):
  # Lista para almacenar los resultados
  results = []

  results = read_to_folder(directory, x, y, m)

  # Ordenar resultados ascendente por la primera columna (número)
  order_index = np.argsort(results[:, 0])
  results = results[order_index]

  # Ajustar los datos usando minimize
  x_data = results[:, 0]
  y_data_b = results[:, 1]
  y_data_g = results[:, 2]
  y_data_r = results[:, 3]
  return x_data, y_data_b, y_data_g, y_data_r

'''

'''

def plot_fit(x_range, x_data, y_data, fit_funct, name, c, coor='cartesian', axis=None):
  if(c == 'blue'):
    index = 0
  elif(c == 'green'):
    index = 1
  else:
    index = 2
  E_min = min(y_data)
  E_max = max(y_data)
  fit_params, covariance = curve_fit(lambda x, s: fit_funct(x, s, E_max, E_min), x_data, y_data)
  s_r = fit_params
  y_fit = ellipse(x_range, s_r, E_max, E_min)
  if(coor == 'cartesian'):
    plt.plot(x_range, y_fit, '-', label=name, color=c)
  elif(coor == 'polar'):
    axis[index].plot(x_range, y_fit, '-', label=name, color=c)
  
'''
'''

def print_results(x_data, y_data_b, y_data_g, y_data_r, zone, fit_function, coor = 'cartesian'):
  file_name = zone + '_' + coor + '_E.pdf'
  if(coor == 'cartesian'):
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data_b, 'o', color='blue', mfc='none')
    plt.plot(x_data, y_data_g, 'o', color='green', mfc='none')
    plt.plot(x_data, y_data_r, 'o', color='red', mfc='none')
    x_range = np.linspace(min(x_data), max(x_data), 100)

    plot_fit(x_range, x_data, y_data_b, fit_function, 'Adjust_b', 'blue')
    plot_fit(x_range, x_data, y_data_g, fit_function, 'Adjust_g', 'green')
    plot_fit(x_range, x_data, y_data_r, fit_function, 'Adjust_r', 'red')
    
    plt.xlabel('Angle[°]')
    plt.title('Point '+zone)
    plt.legend()

  elif(coor == 'polar'):
    figure, axis = plt.subplots(1, 3, subplot_kw=dict(projection="polar"), figsize=(10, 5))
    axis[0].plot(np.deg2rad(x_data), y_data_b, 'o', color='blue', mfc='none')
    axis[1].plot(np.deg2rad(x_data), y_data_g, 'o', color='green', mfc='none')
    axis[2].plot(np.deg2rad(x_data), y_data_r, 'o', color='red', mfc='none')
    x_range = np.linspace(0, 2*np.pi, 1000)

    plot_fit(x_range, np.deg2rad(x_data), y_data_b, fit_function, 'Adjust_b', 'blue', 'polar', axis)
    axis[0].set_title("Blue channel")
    plot_fit(x_range, np.deg2rad(x_data), y_data_g, fit_function, 'Adjust_g', 'green', 'polar', axis)
    axis[1].set_title("Green channel")
    plot_fit(x_range, np.deg2rad(x_data), y_data_r, fit_function, 'Adjust_r', 'red', 'polar', axis)
    axis[2].set_title("Red channel")
    
    figure.suptitle('Point '+zone, fontsize=20)

  plt.grid(True)
  plt.savefig(file_name, format='pdf')
  plt.show()

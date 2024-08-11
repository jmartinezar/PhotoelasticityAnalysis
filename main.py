from ImAn import *

zone = '1'
x = 510
y = 966
m = 20

directory_im = 'data/'

x_data, y_data_b, y_data_g, y_data_r = obtain_data(directory_im, x, y, m)

print_results(x_data, y_data_b, y_data_g, y_data_r, zone, ellipse, 'polar')

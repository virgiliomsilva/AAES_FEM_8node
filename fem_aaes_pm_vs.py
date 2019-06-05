
# coding: utf-8

# In[1]:


# import modules
import numpy as np
import time
from sympy import *
from sympy.interactive import printing
printing.init_printing(use_latex=True)


# In[2]:


# Coordenadas pontos de Gauss + pesos
# |[(2x2)]|
# |[(3x3)]|
# |[(4x4)]|
P = [[  -0.577350269,       -0.577350269]       ,
[       0.5773502692,       -0.577350269]       ,
[       -0.577350269,       0.5773502692]       ,
[       0.5773502692,       0.5773502692]       ,
[       -0.774596669,       -0.774596669]       ,
[       0.0000000000,       -0.774596669]       ,
[       0.7745966692,       -0.774596669]       ,
[       -0.774596669,       0.0000000000]       ,
[       0.0000000000,       0.0000000000]       ,
[       0.7745966692,       0.0000000000]       ,
[       -0.774596669,       0.7745966692]       ,
[       0.0000000000,       0.7745966692]       ,
[       0.7745966692,       0.7745966692]       ,
[       -0.861136311,       -0.861136311]       ,
[       -0.339981043,       -0.861136311]       ,
[       0.3399810436,       -0.861136311]       ,
[       0.8611363116,       -0.861136311]       ,
[       -0.861136311,       -0.339981043]       ,
[       -0.339981043,       -0.339981043]       ,
[       0.3399810436,       -0.339981043]       ,
[       0.8611363116,       -0.339981043]       ,
[       -0.861136311,       0.3399810436]       ,
[       -0.339981043,       0.3399810436]       ,
[       0.3399810436,       0.3399810436]       ,
[       0.8611363116,       0.3399810436]       ,
[       -0.861136311,       0.8611363116]       ,
[       -0.339981043,       0.8611363116]       ,
[       0.3399810436,       0.8611363116]       ,
[       0.8611363116,       0.8611363116]
]
#
W = [[             1,                  1]       ,
[                  1,                  1]       ,
[                  1,                  1]       ,
[                  1,                  1]       ,
[       0.5555555556,       0.5555555556]       ,
[       0.8888888889,       0.5555555556]       ,
[       0.5555555556,       0.5555555556]       ,
[       0.5555555556,       0.8888888889]       ,
[       0.8888888889,       0.8888888889]       ,
[       0.5555555556,       0.8888888889]       ,
[       0.5555555556,       0.5555555556]       ,
[       0.8888888889,       0.5555555556]       ,
[       0.5555555556,       0.5555555556]       ,
[       0.3478548451,       0.3478548451]       ,
[       0.6521451549,       0.3478548451]       ,
[       0.6521451549,       0.3478548451]       ,
[       0.3478548451,       0.3478548451]       ,
[       0.3478548451,       0.6521451549]       ,
[       0.6521451549,       0.6521451549]       ,
[       0.6521451549,       0.6521451549]       ,
[       0.3478548451,       0.6521451549]       ,
[       0.3478548451,       0.6521451549]       ,
[       0.6521451549,       0.6521451549]       ,
[       0.6521451549,       0.6521451549]       ,
[       0.3478548451,       0.6521451549]       ,
[       0.3478548451,       0.3478548451]       ,
[       0.6521451549,       0.3478548451]       ,
[       0.6521451549,       0.3478548451]       ,
[       0.3478548451,       0.3478548451]
]
#
#
#funções
def dNd(s1,s2):
    dN1ds1 = -.5 * (s1 + .5 * s2) * (s2 - 1)
    dN1ds2 = -.5 * (s2 + .5 * s1) * (s1 - 1)
    dN2ds1 = s1 * (s2 - 1)
    dN2ds2 = .5 * (s1 ** 2 - 1)
    dN3ds1 = -.5 * (s1 - .5 * s2) * (s2 - 1)
    dN3ds2 = .5 * (s2 - .5 * s1) * (s1 + 1)
    dN4ds1 = -.5 * (s2 ** 2 -1)
    dN4ds2 = -s2 * (s1 + 1)
    dN5ds1 = .5 * (s1 + .5 * s2) * (s2 + 1)
    dN5ds2 = .5 * (s2 + .5 * s1) * (s1 +1)
    dN6ds1 = -s1 * (s2 + 1)
    dN6ds2 = -.5 * (s1 ** 2 - 1)
    dN7ds1 = .5 * (s1 - .5 * s2) * (s2 + 1)
    dN7ds2 = -.5 * (s2 - .5 * s1) * (s1 - 1)
    dN8ds1 = .5 * (s2 ** 2 -1)
    dN8ds2 = s2 * (s1 - 1)
    dNds = [[dN1ds1, dN1ds2],
           [dN2ds1, dN2ds2],
           [dN3ds1, dN3ds2],
           [dN4ds1, dN4ds2],
           [dN5ds1, dN5ds2],
           [dN6ds1, dN6ds2],
           [dN7ds1, dN7ds2],
           [dN8ds1, dN8ds2]]
    return dNds
#
#
def coorlocal(Qpxx, Qpyy):
    s1, s2 = symbols('s1, s2', real=True)
    
    n1 = .25 * (1 - s1) * (1 - s2) * (-s1 - s2 - 1)
    n2 = .5 * (1 - s1 ** 2) * (1 - s2)
    n3 = .25 * (1 + s1) * (1 - s2) * (s1 - s2 - 1)
    n4 = .5 * (1 + s1) * (1 - s2 ** 2)
    n5 = .25 * (1 + s1) * (1 + s2) * (s1 + s2 - 1)
    n6 = .5 * (1 - s1 ** 2) * (1 + s2)
    n7 = .25 * (1 - s1) * (1 + s2) * (-s1 + s2 - 1)
    n8 = .5 * (1 - s1) * (1 - s2 ** 2)
    
    eq1 = n1 * xbv[0][0] + n2 * xbv[1][0] + n3 * xbv[2][0] + n4 * xbv[3][0] + n5 * xbv[4][0] + n6 * xbv[5][0] + n7 * xbv[6][0] + n8 * xbv[7][0] - Qpxx
    eq2 = n1 * xbv[0][1] + n2 * xbv[1][1] + n3 * xbv[2][1] + n4 * xbv[3][1] + n5 * xbv[4][1] + n6 * xbv[5][1] + n7 * xbv[6][1] + n8 * xbv[7][1] - Qpyy
    
    sol = list(nonlinsolve([eq1, eq2], [s1, s2]))
    return sol
#
#
def Ncalc(s1,s2):
    n1 = .25 * (1 - s1) * (1 - s2) * (-s1 - s2 - 1)
    n2 = .5 * (1 - s1 ** 2) * (1 - s2)
    n3 = .25 * (1 + s1) * (1 - s2) * (s1 - s2 - 1)
    n4 = .5 * (1 + s1) * (1 - s2 ** 2)
    n5 = .25 * (1 + s1) * (1 + s2) * (s1 + s2 - 1)
    n6 = .5 * (1 - s1 ** 2) * (1 + s2)
    n7 = .25 * (1 - s1) * (1 + s2) * (-s1 + s2 - 1)
    n8 = .5 * (1 - s1) * (1 - s2 ** 2)
    Ncal = [[n1, 0],
           [0, n1],
           [n2, 0],
           [0, n2],
           [n3, 0],
           [0, n3],
           [n4, 0],
           [0, n4],
           [n5, 0],
           [0, n5],
           [n6, 0],
           [0, n6],
           [n7, 0],
           [0, n7],
           [n8, 0],
           [0, n8]]
    return Ncal
#
#
def magica(K, delta, f, variables):
    LL = []
    LzL = np.subtract(np.dot(K, delta), f)
    for equation in LzL:
      LL.append(equation[0])
    sol = list(linsolve(LL, variables))
    return sol


# In[3]:


#info

print('O seguinte programa resolve uma "estrutura" quadrilátera')
print('de módulo de rigidez e espessura constante, recorrendo ao')      
print('Método dos Elementos Finitos, de um só elemento, segundo uma')
print('formulação isoparamétrica, recorrendo à discretização de um ')
print('elemento de oito nós, no caso particular do Estado Plano de Tensão.')
print()
print('Para maior facilidade de uso do programa em diferentes casos')
print('não são sugeridas unidades para inserir os dados.')
print('Usar unidades concordantes.')
print()
print('Carregue no Enter para continuar...')
input()
#
#
#                                                                   Data request

# Material Properties
E = float(input('Módulo de Young: '))
v = float(input('Coeficiente de Poisson: '))
print()

# Geometric Properties
print('Numerar os oito nós no sentido anti-horáio')
xbh = np.zeros((2,8))
for j in range(8):
    for i in range(2):
        if i == 0:
            xbh[[i],[j]] = float(input('Coordenada X do ponto ' + str(j + 1) + ': '))            
        else:
            xbh[[i],[j]] = float(input('Coordenada Y do ponto ' + str(j + 1) + ': '))

xbv = np.transpose(xbh)
print()
e = float(input('Espessura do elemento: '))
print()

#Boundary properties
variableCounter = 0
delta = [] #np.zeros((16,1))
for j in range(0,16,2):
    tmp1 = float(input('Deslocamento na direção X no ponto ' + str((j + 1)//2+1) + ' bloqueado? (sim: 0 / não: outro): '))            
    tmp2 = float(input('Deslocamento na direção Y no ponto ' + str((j + 1)//2+1) + ' bloqueado? (sim: 0 / não: outro): '))
    delta.append([symbols('v' + str(variableCounter), real=True) if tmp1 != 0 else 0])
    variableCounter += 1 if tmp1 != 0 else 0
    delta.append([symbols('v' + str(variableCounter), real=True) if tmp2 != 0 else 0])
    variableCounter += 1 if tmp2 != 0 else 0
print()

# Load Properties
load_type = input('Carga discreta (d) ou contínua? (c):')
if load_type == str('d'):
    Qpx = float(input('Coordenada X do ponto de aplicação: '))
    Qpy = float(input('Coordenada Y do ponto de aplicação: '))
    Qvx = float(input('Valor da carga na direção X:'))
    Qvy = float(input('Valor da carga na direção Y:'))
elif load_type ==str('c'):
    print('A numeração dos bordos segue a numeração dos nós, assim sendo, o')
    print('bordo 1 corresponde aos nós 1, 2 e 3, o bordo 2 nós 3, 4 e 5,')
    print('o bordo 3 nós 5, 6 e 7 e o bordo 4 nós 7, 8 e 1')
    bordoin = int(input('Bordo número (1, 2, 3 ou 4): '))
    print('Componente tangencial positiva no sentido anti-horário')
    bt = float(input('Componente tangencial da carga distribuída: '))
    #bt2 = float(input('Componente tangencial da força distribuída: '))
    #bt3 = float(input('Componente tangencial da força distribuída: '))
    print('Componente normal positiva para o interior do elemento')
    bn = float(input('Componente normal da carga distribuída: '))
    #bn2 = float(input('Componente normal da força distribuída: '))
    #bn3 = float(input('Componente normal da força distribuída: '))
print()

#Calc nature
print('Natureza de integração:')
print('Integração Reduzida       (2x2) - r')
print('Integração Completa       (3x3) - c')
#print('Integração Superabundante (4x4) - s')
n = input('                                  ')
#
start_time = time.time()




#                                                                     Constantes 
if n == str('r'):
    a = 0                     
    b = 4       
    c = 2
elif n == str('c'):
    a = 4
    b = 13
    c = 7
#elif n == str('s'):
#    a = 13
#    b = 29
#    c = 17


#matriz constitutiva ou de elasticidade
D = np.array([[    E / (1 - v ** 2), E * v / (1 - v ** 2),                0],
              [E * v / (1 - v ** 2),     E / (1 - v ** 2),                0],
              [                   0,                    0, E / (2 * (1 + v))]])


#                                                             Stiffness matrix K
K = np.zeros((16,16))
B = np.zeros((3,16))

for i in range(a,b):
    dNds = dNd(P[i][0],P[i][1]) #derivadas de funções de forma calculadas nos pontos de gauss
    
    J = np.dot(xbh, dNds)
    Jdet = np.linalg.det(J)
    Jinv = np.linalg.inv(J)
    dNdx = np.dot(dNds, Jinv)

    for u in range(8):
        B[[0],[2 * u]] = dNdx[[u],[0]]
        B[[1],[2 * u + 1]] = dNdx[[u],[1]]
        B[[2],[2 * u]] = dNdx[[u],[1]]
        B[[2],[2 * u + 1]] = dNdx[[u],[0]]
    Bt = np.transpose(B)

    Kaux = np.dot((np.dot(Bt, D)), B) * e * Jdet * W[i][0] * W[i][1]
    K += Kaux
    
#                                                                          Loads
F = np.zeros((16,1))

if load_type == str('d'):          #point loads
    sps = coorlocal(Qpx, Qpy)

    Ncalcu = Ncalc(sps[0][0], sps[0][1])
    Qvec = np.array([[Qvx],[Qvy]])
    F = np.dot(Ncalcu,Qvec)

elif load_type == str('c'):       #distributed loads
    Faux = np.zeros((6,1))
    pDistri = np.array([[bt, bt, bt], [bn, bn, bn]])
    
    if bordoin == int(1):
        c1 = 0
        c2 = 1
        c3 = 2
    elif bordoin == int(2):
        c1 = 2 
        c2 = 3
        c3 = 4    
    elif bordoin == int(3):
        c1 = 4 
        c2 = 5
        c3 = 6
    elif bordoin == int(4):
        c1 = 6
        c2 = 7
        c3 = 0
    
    
    for i in range(a,c):
        s = P[i][0]
        
        Ni1 = .5 * (s ** 2 - s)
        Ni2 = 1 - s ** 2
        Ni3 = .5 * (s ** 2 + s)
        
        NDS1 = s - .5
        NDS2 = -2 * s
        NDS3 = s + .5   
               
        dxids1 = NDS1 * xbv[c1][0] + NDS2 * xbv[c2][0] + NDS3 * xbv[c3][0]
        dxjds1 = NDS1 * xbv[c1][1] + NDS2 * xbv[c2][1] + NDS3 * xbv[c3][1]    

        Tt = [[dxids1, -dxjds1],[dxjds1, dxids1]]
        Nv = np.array([[Ni1], [Ni2], [Ni3]])
        NN = np.array([[Ni1, 0],[0, Ni1],[Ni2, 0],[0, Ni2],[Ni3, 0],[0, Ni3]])  
            
        Faux2 = np.dot(np.dot(np.dot(NN, Tt), pDistri),Nv) * W[i][0]
        Faux += Faux2
    
    if bordoin == int(1):
        F[[0][0]] = Faux[[0],[0]]
        F[[1][0]] = Faux[[1],[0]]
        F[[2][0]] = Faux[[2],[0]]
        F[[3][0]] = Faux[[3],[0]]
        F[[4][0]] = Faux[[4],[0]]
        F[[5][0]] = Faux[[5],[0]]
        
    elif bordoin == int(2):
        F[[4][0]] = Faux[[0],[0]]
        F[[5][0]] = Faux[[1],[0]]
        F[[6][0]] = Faux[[2],[0]]
        F[[7][0]] = Faux[[3],[0]]
        F[[8][0]] = Faux[[4],[0]]
        F[[9][0]] = Faux[[5],[0]]
        
    elif bordoin == int(3):
        F[[8][0]] = Faux[[0],[0]]
        F[[9][0]] = Faux[[1],[0]]
        F[[10][0]] = Faux[[2],[0]]
        F[[11][0]] = Faux[[3],[0]]
        F[[12][0]] = Faux[[4],[0]]
        F[[13][0]] = Faux[[5],[0]]
        
    elif bordoin == int(4):
        F[[12][0]] = Faux[[0],[0]]
        F[[13][0]] = Faux[[1],[0]]
        F[[14][0]] = Faux[[2],[0]]
        F[[15][0]] = Faux[[3],[0]]
        F[[0][0]] = Faux[[4],[0]]
        F[[1][0]] = Faux[[5],[0]]

        
#                                                                         Solver
f = []
variables = []
for i in range(len(delta)):
  if delta[i][0] == 0:
    symbol = symbols('f' + str(len(f)), real=True)
    f.append([symbol])
    variables.append(symbol)
    
  else:
    f.append([F[i][0]])
    variables.append(delta[i][0])

sol = magica(K,delta,f,variables)


dict = {}
for i in range(len(sol[0])):
  dict[variables[i]] = sol[0][i]
  
ResF = []
for v in f:
  if type(v[0]) != Symbol:
    ResF.append(v[0])
    
  else:
    ResF.append(dict[v[0]])

ResDelta = []
for v in delta:
  if type(v[0]) != Symbol:
    ResDelta.append(v[0])
    
  else:
    ResDelta.append(dict[v[0]])


#                                                                        Results
total_time = round(time.time() - start_time,2)
KK = Matrix(K)
DD = Matrix(ResDelta)
FF = Matrix(ResF)
print()
print()
print('Resultados')
print()
print('Tempo de cálculo:',total_time, 'segundos')
print()
print('Deslocamentos nodais')
print(ResDelta)
#display(DD)
print()
print('Forças nodais')
print(ResF)
#display(FF)
print()
print('Matriz Rigidez')
print(K)
#display(KK)


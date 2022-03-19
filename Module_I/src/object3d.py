import numpy as np
from .vector import Vector3 as vector
from math import sin, cos

class Object3D:
    """Класс для формирования точечных 3D фигур"""
    OBJ_POINT = 0
    OBJ_CUBE = 1
    OBJ_MISiS = 2

    def __init__(self, pos : vector, eulerRot : vector, type : int = 0, width = 1.0, height = 1.0, length = 1.0):
        self.rootPos = pos                              # Координаты центра объекта
        self.rot = eulerRot                             # Угол поворота
        self.type = type                                # Тип отображаемой фигуры
        self.w, self.h, self.l = width,height,length    # Ширина, Высота и Длина
        self.calcByType()                               # Функция для выполнения расчёта координат точек и связей формирующих фигуру

    def setRotation(self, eurlerRot : vector):
        self.rot = eurlerRot

    def calcByType(self):
        self.vectors = []
        self.links = []
        if self.type == Object3D.OBJ_CUBE:
            for x in [-0.5,0.5]: # Генерация точек всех углов в нужном порядке
                for y in [-0.5,0.5]:
                    for z in [-0.5,0.5]:
                        z *= round(np.sign(y))
                        self.vectors.append(vector(x*self.w,y*self.l,z*self.h))
            for i in range(4): 
                self.links.append( (i, i+4) ) # Настройка связей в формате (ОТ,ДО) для соединения линиями в нужной последовательности..
                self.links.append( (i, (i+1)%4) )
                self.links.append( tuple(i+4 for i in self.links[-1] ) )
        elif self.type == Object3D.OBJ_MISiS:
            charM = ( vector(-0.4,0,-0.5), vector(-0.4,0,0.5), vector(0.0,0,0.0), vector(0.4,0,0.5), vector(0.4,0,-0.5) )
            charIb = ( vector(-0.4,0,0.5), vector(-0.4,0,-0.5), vector(0.4,0,0.5), vector(0.4,0,-0.5) )
            charIs = ( vector(-0.4,0,0.2), vector(-0.4,0,-0.5), vector(0.4,0,0.2), vector(0.4,0,-0.5) )
            charS = ( vector(0.4,0,0.5), vector(-0.3,0,0.5), vector(-0.4,0,0.4), vector(-0.4,0,-0.4), vector(-0.3,0,-0.5), vector(0.4,0,-0.5) )
            text = (charM, charIb, charS, charIs, charS)
            
            lt = len(text)
            for o in range(lt):
                last = len(self.vectors)
                for p in text[o]:
                    self.vectors.append( vector( (p.x + (-3.6+1.2*o))/lt*self.w, p.y/lt*self.l, p.z/lt*self.h) )
                for i in range(last, len(self.vectors)-1):
                    self.links.append( (i,i+1) )
        else: # Object3D.OBJ_POINT
            self.vectors.append(self.rootPos)

    def get(self): 
        calcvectors = []

        R = lambda a, b, c: np.array([
            [ cos(a)*cos(b), cos(a)*sin(b)*sin(c)-sin(a)*cos(c), cos(a)*sin(b)*cos(c)+sin(a)*sin(c) ],
            [ sin(a)*cos(b), sin(a)*sin(b)*sin(c)+cos(a)*cos(c), sin(a)*sin(b)*cos(c)-cos(a)*sin(c) ],
            [ -sin(b), cos(b)*sin(c), cos(b)*cos(c) ]
        ])
        R = R(self.rot.x, self.rot.y, self.rot.z)

        for p in self.vectors:
            a = self.rootPos.vec + R.dot(p.vec)
            calcvectors.append(vector(
                tuple(a.T[0])
            ))
        return (calcvectors, self.links)
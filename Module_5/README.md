# Список заданий для выполнения
В папке data/tram располагаются клипы. C использованием преобразования Хафа и методов проективной геометрии определите на изображении:<br />
<br />
1 вертикальные линии, вычислите 3d координаты их нижней точки и нанесите их на локальную карту;<br />
2 трамвайные линии, зная ширину трамвайных путей;<br />
3 линию горизонта на основе vanisihg point;<br />
**Решение:**<br />
1. Найти границы на кадре детектором Канни
2. На их основе найти линии преобразованием Хаффа
3. Отсеить линии, не принадлежащие рельсам
4. Найти все точки пересечения линий с потенциальной линией горизонта
5. Проведение линии горизонта через самую насыщенную пересечениями точку

**Результат:**<br />
Отрисованные на кадре: синие линии Хаффа; зелёная, строго горизонтальная линия горизонта; точка пересечения<br />
<br />
4 проекции трамвайных линий и оцените точный угол наклона камеры на основе корреляционного соответствия двух пар линий: трамвайных путей и проекций трамвайных путей в кадр;<br />
5 с использованием преобразования Хафа для окружностей найдите на изображении светофоры;<br />

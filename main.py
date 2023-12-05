import numpy
import interpolate as interpol
import matplotlib.pyplot as plt
import scipy.interpolate as scp


if __name__ == '__main__':
    data = numpy.array([[1.,3.],[2.,1.],[3.5,4.],[5.,0.],[6.,0.5],[9.,-2.],[9.5,-3.]])
    start = 0
    koniec = 10
    krok = 1000
    x = numpy.linspace(start, koniec, krok)
    print(x)
    skl_3 = interpol.Interplacja_sklejana_3st(data, start, koniec, krok).Si
    skl_1 = interpol.Sklejana_stopnia_pierwszego(data, start, koniec, krok).sklej
    lagrange = interpol.Lagrange(data,start, koniec, krok).dzialaXDDDD
    plt.plot(x, skl_3, label = 'Interpolacja sklejana trzeciego stopnia')
    plt.plot(x, skl_1, label='Interpolacja sklejana pierwszego stopnia')
    plt.plot(x, lagrange, label="Interpolacja Lagrange'a")
    plt.legend()
    plt.show()

    kubara_cyganskie = scp.CubicSpline(data[:, 0], data[:, 1])
    plt.plot(x, kubara_cyganskie(x), label = 'Cubic Spline!!!!!!!!!!')
    plt.plot(x, skl_3, label = 'Trzeciego stopnia chrumk!!!!')
    plt.legend()
    plt.show()


''' W warunkach bez węzła Cubic Spline jest skonstruowany w taki sposób, że trzecia pochodna jest ciągła w punktach końcowych przedziału.
Oznacza to, że Cubic Spline ma ciągłą krzywiznę w punktach granicznych, dzięki czemu jest gładki na obu końcach.
Warunki bez węzła zasadniczo zapewniają, że Cubic Spline nie ma punktów przegięcia w punktach końcowych.'''

'''W normalnych warunkach brzegowych druga pochodna w punktach końcowych jest ustawiona na zero.
Oznacza to, że funkcja sklejna trzeciego stopnia jest liniowa poza pierwszym i ostatnim interwałem, co czyni ją naturalnym rozszerzeniem interpolacji liniowej w granicach.
Warunki naturalne zapewniają płynniejsze zachowanie w punktach końcowych, ale nie wymuszają ciągłości trzeciej pochodnej.'''




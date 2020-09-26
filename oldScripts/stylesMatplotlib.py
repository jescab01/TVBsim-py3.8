# From https://pybonacci.org/2015/01/25/realmente-es-feo-matplotlib/

# My preferred styles: pastelJescab01 | seaborn | seaborn-whitegrid | ggplot | seaborn-paper

print('Estilos disponibles: ', plt.style.available)
estilo = numpy.random.choice(plt.style.available)
print('Vamos a usar el estilo ', estilo)
plt.style.use(estilo)
# la misma gráfica que antes
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(numpy.random.randn(25), label = 'random')
ax.legend()
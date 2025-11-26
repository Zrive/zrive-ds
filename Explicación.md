Lo que observamos en nuestro modelo es que este aprende "demasiado rápido", creando por lo tanto un
gran overfitting en "pocas tiradas". Es por ello que reducimos el learning rate, así cómo posibles 
pruebas de pruning a los árboles. A su vez, observamos ciertas casuísticas que pueden tergiversar el aprendizaje del modelo, y por lo tanto la interpretación de sus resultados y su posterior puesta en producción. Estas son principalmente dos:

- Estamos comparando el sp500 con dividendos con respectos a los stocks en concreto con dividendo.  
  Es decir, se podría decir que estamos comparando peras con manzanas, a no ser que seamos fieles creyentes de la Teoría Eficiente de los Mercados, la cuál no daría ni mucho menos por hecho. Creo que es mucho más recomendable usar en ambos casos dividendos, sería más realista y nos evitamos ciertos problemas.

- Estamos usando un "Execution_date" muy lejano a las entregas de resultados de las empresas, con   lo cual, estamos reaccionando muy tarde con respecto a lo que lo hace el mercado, implicando esto una muy posible perdidad de rentabilidad, ya que no estamos actuando cuando tenemos información.

- Quizás podamos tener cierto survivorship bias.

Por último también observamos como la variable "Close_0" y "Close_sp500_0" tienen un poder de predicción muy grande, aunque quizás no tenga mucho fundamento teórico/práctico. En especial, en el primer caso, es muy probable que tengamos cierto "data leakage" reflejado en aquellas empresas que han tenido un split/contrasplit. Podríamos optar por eliminar esta variable, o eliminar aquellas filas que hayan sufrido un split/contrasplit.
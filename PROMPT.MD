en el paso 5 del notebook me salta este error cuando quiero que se generen las figuras:

Generando figuras en /content/generador-academico-rag/figs
  [ok] notas_distribucion.png
  [ok] avance_anual.png
  [ok] radar_areas.png
  [ok] clustering_perfiles.png
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_522/745978796.py in <cell line: 0>()
----> 1 figuras.generar_todo()  # regenera los PNG con estilo Graphite & Cobalt
      2 for png in ["notas_distribucion.png", "avance_anual.png", "radar_areas.png"]:
      3     display(Image(filename=str(figuras.FIGS / png)))

5 frames
/usr/local/lib/python3.12/dist-packages/sklearn/manifold/_t_sne.py in _check_params_vs_input(self, X)
    860     def _check_params_vs_input(self, X):
    861         if self.perplexity >= X.shape[0]:
--> 862             raise ValueError("perplexity must be less than n_samples")
    863 
    864     def _fit(self, X, skip_num_points=0):

ValueError: perplexity must be less than n_samples


Corregi eso y verifica tambien los subsiguientes pasos para asegurarte de que todo ande correctamente.

Luego de corregir eso, te pido lo siguiente: simplificar proyecto (borrar archivos redundantes, innecesarios, etc). Simplificar el readme, especialmente en la parte de instrucciones de uso (debe estar como primera opcion el uso de colab, y debe ser de manera sencilla y directa la explicacion de como usarlo, y debe aclarar que eso es si se quiere usar la web app. que en caso de querer probar y correr punta a punta el generador directo en colab, usar el notebook con sus archivos relevantes y listo). Y taqmvbien que el notebook sea mas directo y sintetico, especialmente al principio, aclarando dos cosas importantses (no se si es posible resaltar en color pero intenta): advetir que el runtime lo haga con t4 gpu, y que tambien que sea PACIENTE, porque tarda unos minutos en ejecutarse todo, Y que este atento tambien que en el primer paso te pide insertar el ZIP con el src y todo eso 8o, alternativamnte, puede subirse al drive, pero especificar bien que debe subirse exactamente! y advertir en emsnaje de erorr en caso de que falte algo, de manera clara).

ver tambien por que la metrica de faithfullness da casi lo mismo en ambos casos, siendo que en uno si usa RAG y en el otro no! a qué se debe que den ambos cercano a 0.5? (métrica llm as a judge)

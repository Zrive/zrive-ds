# DOCUMENTO DE REQUISITOS DEL PROYECTO - TRD
import pandas as pd

# ------------------------------------------------------------------------------------------------------------
# 1. Objetivos y contexto.
# Nuestros clientes realizan compras en nuestra tienda de comestibles a través de la aplicación.
# Algunos productos queremos promocionarlos: porque están próximos a caducar o para aumentar la cuota de mercado.

# Enviar notificaciones push a usuarios es una forma eficaz de impulsar ventas y ofrecer descuentos.
# Sin embargo, demasiadas notificaciones pueden generar insatisfacción y abandono de la app.

# Construir un sistema basado en un modelo predictivo que permita identificar usuarios altamente propensos a
# interesarse por un producto y enviarles notificaciones push personalizadas.
# -------------------------------------------------------------------------------------------------------------


# Cargar dataset
feature_frame_filtered = pd.read_csv(r"C:\Users\Lucia\PycharmProjects\zrive-ds\src\module_3\data\feature_frame_filtered.csv")

# Número de productos únicos
num_productos_unicos = feature_frame_filtered['product_type'].nunique()
print(f"Hay {num_productos_unicos} productos únicos en el dataset.") # hay 62 productos

# =========================================
# Clase para gestionar notificaciones push
# =========================================
class NotificacionPush:
    """
    Clase para gestionar notificaciones push a usuarios propensos usando un modelo predictivo.
    """

    def __init__(self, dataframe, umbral=0.5):
        self.dataframe = dataframe
        self.umbral = umbral

    def seleccionar_producto(self):
        productos = self.dataframe['product_type'].value_counts().index.tolist()
        print("Seleccione un producto de la lista (ingrese el número):")
        for i, p in enumerate(productos[:62], 1):
            print(f"{i}. {p}")

        eleccion = input("Número del producto o escriba el nombre: ")

        if eleccion.isdigit():
            eleccion = int(eleccion)
            if 1 <= eleccion <= len(productos[:62]):
                producto_seleccionado = productos[eleccion - 1]
                print(f"Has seleccionado: {producto_seleccionado}")
                return producto_seleccionado

        if eleccion in productos:
            print(f"Has seleccionado: {eleccion}")
            return eleccion

        print("Producto no encontrado. Intente de nuevo.")
        return self.seleccionar_producto()

    def seleccionar_usuarios(self, producto):
        df_producto = self.dataframe[self.dataframe['product_type'] == producto]
        usuarios_objetivo = df_producto[df_producto['outcome'] >= self.umbral]
        return usuarios_objetivo

    def lanzar_notificacion(self, producto, mensaje):
        usuarios_objetivo = self.seleccionar_usuarios(producto)
        total_usuarios_producto = self.dataframe[self.dataframe['product_type'] == producto].shape[0]

        # Enviar notificación simulada
        for index, usuario in usuarios_objetivo.iterrows():
            print(f"Enviando notificación a user_id {usuario['user_id']}: {mensaje}")

        # Métricas de impacto esperado
        print("\n--- Métricas simuladas de impacto ---")
        print(f"Producto promocionado: {producto}")
        print(f"Número de usuarios que recibirían la notificación: {len(usuarios_objetivo)}")
        print(f"Total de usuarios que compraron el producto: {total_usuarios_producto}")
        if total_usuarios_producto > 0:
            porcentaje_cubierto = len(usuarios_objetivo) / total_usuarios_producto * 100
            print(f"Porcentaje de cobertura del producto: {porcentaje_cubierto:.2f}%")
        else:
            print("No hay usuarios para este producto.")


# =========================================
# Ejecución de la PoC
# =========================================
if __name__ == "__main__":
    notificacion = NotificacionPush(dataframe=feature_frame_filtered, umbral=0.6)

    # Selección interactiva de producto
    producto = notificacion.seleccionar_producto()

    # Mensaje de la notificación
    mensaje = f"¡Oferta especial! Compra {producto} y recibe un descuento exclusivo."

    # Lanzar notificación y mostrar métricas de impacto
    notificacion.lanzar_notificacion(producto, mensaje)


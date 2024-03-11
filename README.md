# zrive-ds
Este repositorio contiene la estructura que utilizaremos para el programa Zrive Applied Data Science (https://zriveapp.com/cursos/zrive-applied-data-science).

## Set up
Para asegurar consistencia utilizaremos todos **python version = 3.11.0**, además usaremos `pyenv` para la gestión de versiones de python y `poetry` para gestionar virtualenvs y dependencias. Estas tecnologías son una elección en cierta medida arbitraria, aunque las herramientas utilizadas son ampliamente utilizadas, hay otras opciones como `venv`, `virtualenv` o `pipenv` para virtualenvs, o conda para gestión de paquetes, que también se utilizan de forma extensa.

**RECORDAD: Es importante no utilizar el python del sistema ya que acabaremos teniendo conflictos entre proyectos muy rápido. Por eso, utilizaremos siempre un virtualenv separado para cada proyecto.**

Para configurar nuestro entorno de trabajo primero instalamos python 3.11.0 en el ordenador usando pyenv `pyenv install 3.11.0`. 
Después, una vez estemos en el repositorio de trabajo (tras haber hecho un fork del repositorio original), nos aseguramos de que estamos utilizando ese python localmente `pyenv local 3.11.0`. Podemos comprobar que todo está correcto mediante el comando `pyenv versions`, que nos debería mostrar 3.11.0 con un asterisco que denota que es la versión utilizada en el directorio actual. IMPORTANTE: si utilizáis ubuntu con WSL es posible que poetry no utilice la versión local de `pyenv` y sea necesario forzarlo con `poetry env use 3.11.0` [ver aquí](https://stackoverflow.com/questions/70950511/using-poetry-with-pyenv-and-having-python-version-issues).

Una vez asegurada que la versión de python es correcta pasamos a instalar las dependencias iniciales definidas en `pyproject.toml` mediante el comando `poetry install`. Una vez instaladas, si necesitamos añadir una nueva dependencia podemos hacerlo con `poetry add <paquete>`.

Además, para garantizar la estandarización del código utilizamos:
1. `Black`: formatter no configurable que fuerza que el código tome el mismo formato en todos los proyectos.
2. `Flake8`: is a code linter. It warns you of syntax errors, possible bugs, stylistic errors, etc.

Finalmente, también tenemos instalado `mypy` para checkear tipos en nuestro código, sin embargo no lo tenemos en uso activo en `make test` ya que requiere cierta configuración. _Mypy is a static type checker for Python). Python is a dynamic language, so usually you'll only see errors in your code when you attempt to run it. Mypy is a static checker, so it finds bugs in your programs without even running them!_

## Repo structure
Utilizaremos la estructura por defecto que genera poetry al crear un nuevo proyecto y separamos el código de cada módulo y sus tests en una subcarpeta para facilitar la revisión del mismo. Dentro del `zrive-ds` añadiremos todo el código, en forma de jupyter notebooks o scripts. En tests, añadiremos tests unitarios cuando se pidan que permitan testear el código que hemos desarrollado para "garantizar" su correcto funcionamiento.

Utilizar `make lint` y `make test`, definidos en la Makefile para estandarizar el linting y asegurar que todos los tests corren. Esto facilitará también la revisión del código.
```
zrive-ds
├── pyproject.toml
├── README.md
├── src
│   ├── __init__.py
│   ├── module 1
│   ├── module 2
│   ├── module 3
│   ├── module 4
│   ├── module 5
│   └── module 6
└── tests
    └── __init__.py
    ├── module 1
    ├── module 2
    ├── module 3
    ├── module 4
    ├── module 5
    └── module 6
```


## Data
Todos los datos estarán disponibles en s3, accesible mediante el uso de access keys que os facilitaré yo para que guardéis en vuestro `.env` para evitar subirlo al repositario en plain text (compartiré el enlace a 1password para obtener los access keys).

Una vez tengais las access keys podeis conectaros utilizando AWS CLI con las siguientes instrucciones:
1. `aws configure` -> tras la cual, solo tenéis que añadir las access keys, lo demas lo podéis dejar por defecto.
2. `aws s3 ls s3://zrive-ds-data/ --recursive` -> Para mostrar los datos en s3

Los datos encontraréis disponibles en s3 son:
1. `groceries/sampled-datasets/`:
    - orders.parquet: An order history of customers. Each row is an order and the item_ids for the order are stored as a list in the item_ids column
    - regulars.parquet: We allow users to specify items that they wish to buy regularly. This data gives the items each user has asked to get regularly, along with when they input that information.
    - abandoned_cart.parquet: If a user has added items to their basket but not bought them, we capture that information. Items that were abandoned are stored as a list in item_ids.
    - inventory.parquet: Some information about each item_id that may prove useful for your model
    - users.parquer: Information about users that may be useful for the model.

2. `groceries/box_builder_dataset/` -> full-size dataset donde cada "instance" corresponde al triplet (order, user, product):
    - El dataset solo incluye una selección de productos que excluye el long tail de productos. 
    - Este dataset solo incluye "repeating orders" (excluye las primeras ordenes). 
    - La etiqueta "outcome" binaria es si el producto fue comprado o no en esa orden.
    - Este dataset está pensado para entrenar un modelo que permita predecir que productos poner en la siguiente cesta antes de que llegue el usuario.
    - Se puede utilizar también para simular otros problemas:
        - Serie temporal de ventas de productos.
        - Probabilidad de que un usuario compre X dias tras la compra previa.
        - Predecir el valor (GBP) de la siguiente compra.


La primera vez que vayais a trabajar con los datos, debeis descargarlos utilizando python `boto3` o `AWS CLI` y a partir de ahí guardarlos en local para evitar tener que descargarlos cada vez (IMPORTANTE: Se considera una buena práctica no añadir datos a vuestros repositorios de código ya que eso haría el repositorio muy pesado y todas las operaciones de version control muy lentas.)


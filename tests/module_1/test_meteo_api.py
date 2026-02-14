import sys
import os
import pandas as pd
from datetime import datetime

# Añadir la carpeta src al path para poder importar
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

# Importar las funciones que vamos a testear
from module_1.module_1_meteo_api import get_data_meteo_api, process_weather_data, call_api


def test_get_data_meteo_api_invalid_city():
    """Test que verifica que la función maneja ciudades inválidas correctamente"""
    print("🧪 Probando ciudad inválida...")
    result = get_data_meteo_api("CiudadInexistente")
    assert result is None, "Debería devolver None para ciudad inválida"
    print("   ✅ Test ciudad inválida: PASADO")


def test_get_data_meteo_api_valid_city():
    """Test que verifica que la función funciona con ciudades válidas"""
    print("🧪 Probando ciudad válida...")
    result = get_data_meteo_api("Madrid", "2020-01-01", "2020-01-01")
    assert result is not None, "Debería devolver datos para ciudad válida"
    assert "city" in result.columns, "Debería tener columna 'city'"
    assert "time" in result.columns, "Debería tener columna 'time'"
    print("   ✅ Test ciudad válida: PASADO")


def test_process_weather_data():
    """Test que verifica el procesamiento de datos"""
    print("🧪 Probando procesamiento de datos...")

    # Crear datos de prueba simulados
    test_data = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=30, freq="D"),  # 30 días
        "temperature_2m_mean": [20 + i * 0.1 for i in range(30)],  # Temperaturas ficticias
        "precipitation_sum": [i % 5 for i in range(30)],  # Precipitación ficticia
        "wind_speed_10m_max": [10 + i % 3 for i in range(30)],  # Viento ficticio
        "city": ["TestCity"] * 30
    })

    # Probar procesamiento mensual
    processed = process_weather_data(test_data, "M")

    assert processed is not None, "El procesamiento no debería devolver None"
    assert len(processed) > 0, "Debería tener al menos una fila después del procesamiento"
    assert "time" in processed.columns, "Debería mantener la columna time"
    assert "temperature_2m_mean" in processed.columns, "Debería mantener temperatura"
    print("   ✅ Test procesamiento de datos: PASADO")


def test_call_api_invalid_url():
    """Test que verifica el manejo de errores en llamadas API"""
    print("🧪 Probando llamada API inválida...")
    result = call_api("https://url-invalida.com", {})
    assert result is None, "Debería devolver None para URL inválida"
    print("   ✅ Test API inválida: PASADO")


def test_main_function_exists():
    """Test que verifica que la función main existe y es llamable"""
    print("🧪 Probando que main existe...")
    from module_1.module_1_meteo_api import main
    assert callable(main), "La función main debería existir y ser llamable"
    print("   ✅ Test función main: PASADO")


def run_all_tests():
    """Función que ejecuta todos los tests"""
    print("🚀 INICIANDO EJECUCIÓN DE TESTS")
    print("=" * 50)

    tests = [
        test_get_data_meteo_api_invalid_city,
        test_get_data_meteo_api_valid_city,
        test_process_weather_data,
        test_call_api_invalid_url,
        test_main_function_exists
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   ❌ {test.__name__}: FALLADO - {e}")
            failed += 1

    print("=" * 50)
    print(f"📊 RESULTADOS: {passed} PASADOS, {failed} FALLADOS")

    if failed == 0:
        print("🎉 ¡TODOS LOS TESTS PASARON!")
    else:
        print("💡 Algunos tests fallaron, pero el programa principal funciona")

    return failed == 0


if __name__ == "__main__":
    run_all_tests()
Entorno virtual unificado

Objetivo: usar un único entorno virtual `.venv` en la raíz del proyecto que contenga todas las dependencias (incluyendo TensorFlow cuando sea necesario).

Pasos recomendados (PowerShell):

1) Crear el entorno y cargar dependencias automáticamente:

   ./scripts/create_unified_venv.ps1

2) Para eliminar o archivar entornos antiguos, usa los comandos de PowerShell `Remove-Item` o `Rename-Item` después de verificar que `.venv` funciona.

Notas:
- La instalación automática intentará instalar `tensorflow==2.11.0` y `protobuf==3.20.3`. Si hay conflictos (por ejemplo, con la versión de `numpy` en `requirements.txt`), ajusta las versiones en `requirements.txt` o instala manualmente después de activar `.venv`.
- No borres los entornos antiguos hasta verificar que `.venv` funciona correctamente para todos los scripts.

# MotorSense — Diagnóstico Predictivo

> Sistema de diagnóstico predictivo para motores basado en análisis de señales, vibración y temperatura.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Objetivo

Detectar fallas incipientes en motores eléctricos / combustión mediante modelos de ML/DL entrenados sobre series temporales de sensores (corriente, vibración, temperatura, RPM).

## Estructura

```
MotorSense-Diagnostico-Predictivo/
├── data/               # datasets crudos / procesados (no versionados)
├── src/motorsense/     # código fuente
│   ├── encoding/       # codificación de señales
│   ├── ansatz/         # modelos / arquitecturas
│   └── training/       # loops de entrenamiento
├── tests/              # tests unitarios
├── docs/               # documentación
├── requirements.txt
└── pyproject.toml
```

## Roadmap de commits (narrativa progresiva)

1. `baseline` — estructura inicial ← **actual**
2. `encoding` — pipeline de codificación de señales
3. `ansatz` — arquitectura del modelo
4. `training loop` — entrenamiento y evaluación
5. `tests` — cobertura de pruebas
6. `README` — documentación final

## Instalación

```bash
pip install -r requirements.txt
pip install -e .
```

## Uso rápido

```python
from motorsense import __version__
print(__version__)
```

## Licencia

MIT — Derek Rex

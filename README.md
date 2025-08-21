# 🍷 Análisis PCA de Vinos: Álgebra Lineal para IA

*url git: https://github.com/Carlitos4586/wine-pca-analysis/tree/main*

## 📋 Descripción

Implementación educativa de **Análisis de Componentes Principales (PCA)** aplicando conceptos fundamentales de álgebra lineal para demostrar su uso en Inteligencia Artificial. Reducimos 13 variables químicas de vinos a 2 componentes principales manteniendo ~75% de la información.

## 🎯 Objetivos

- **Aplicar álgebra lineal:** Matrices, eigenvalues, eigenvectors, transformaciones
- **Reducir dimensionalidad:** De 13D → 2D con mínima pérdida de información
- **Visualizar patrones:** Separar automáticamente 3 clases de vino
- **Conectar con IA:** Demostrar aplicaciones en machine learning moderno

## 📊 Dataset

**UCI Wine Dataset:** 178 vinos con 13 características químicas (alcohol, acidez, magnesio, etc.) clasificados en 3 cultivares.

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/wine-pca-analysis.git
cd wine-pca-analysis

# Crear entorno conda
conda create --name wine-pca python=3.9
conda activate wine-pca

# Instalar dependencias
conda install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## 💻 Uso

### Script Python Completo
```bash
cd src
python wine_analysis.py
```
**Genera:** 5 visualizaciones + informe final automáticamente

### Notebook Educativo
```bash
jupyter notebook notebooks/wine_pca_tutorial.ipynb
```
**Incluye:** 17 celdas paso a paso con explicaciones matemáticas detalladas

## 📁 Estructura del Proyecto

```
wine-pca-analysis/
├── src/
│   └── wine_analysis.py          # Script completo con análisis avanzado
├── notebooks/
│   └── wine_pca_tutorial.ipynb   # Tutorial educativo paso a paso
├── results/                      # Visualizaciones generadas automáticamente
├── docs/                         # Documentación adicional
└── README.md                     # Este archivo
```

## 🧮 Conceptos de Álgebra Lineal Aplicados

| Concepto | Aplicación | Resultado |
|----------|------------|-----------|
| **Matrices** | Representación de datos | 178×13 → estructura organizada |
| **Transformaciones lineales** | Estandarización | Media=0, std=1 |
| **Producto matricial** | Matriz de covarianza | Relaciones entre variables |
| **Eigenvalues/Eigenvectors** | Componentes principales | Direcciones de máxima varianza |
| **Cambio de base** | Proyección PCA | Nuevos ejes ortogonales |
| **Reducción dimensional** | 13D → 2D | 75% información preservada |

## 📈 Resultados

- ✅ **Reducción exitosa:** 84.6% menos dimensiones
- ✅ **Información preservada:** ~75% de varianza total
- ✅ **Separación de clases:** 90%+ precisión visual
- ✅ **Visualización clara:** Patrones antes invisibles

## 🤖 Aplicaciones en IA

- **Computer Vision:** Eigenfaces, compresión de imágenes
- **Machine Learning:** Preprocesamiento, reducción de overfitting
- **Deep Learning:** Inicialización de autoencoders
- **Big Data:** Análisis eficiente de datasets masivos
- **Sistemas de recomendación:** Factorización matricial

## 🔧 Diferencias entre Archivos

| Archivo | Propósito | Características |
|---------|-----------|-----------------|
| `wine_analysis.py` | Análisis completo | 6 visualizaciones, funciones avanzadas |
| `wine_pca_tutorial.ipynb` | Educativo | 17 celdas, explicaciones paso a paso |

## 📚 Referencias Principales

- Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review. *Phil. Trans. R. Soc. A*, 374(2065).
- Strang, G. (2019). *Linear algebra and learning from data*. Wellesley-Cambridge Press.
- UCI ML Repository: Wine Dataset. http://archive.ics.uci.edu/ml

## 👨‍💻 Autor

**Carlos A. Bermudez I.**  
Curso: Matemáticas para IA - Álgebra Lineal  
Fecha: Agosto 2025

## 📄 Licencia

Este proyecto tiene fines educativos. Dataset UCI bajo licencia CC BY 4.0.

---

**🎓 Proyecto académico demostrando la aplicación práctica del álgebra lineal en Inteligencia Artificial**
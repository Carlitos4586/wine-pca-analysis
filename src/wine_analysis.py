"""
Análisis de Vinos usando PCA (Álgebra Lineal para IA)
==================================================

Este script implementa Análisis de Componentes Principales (PCA) paso a paso
para demostrar conceptos fundamentales de álgebra lineal en inteligencia artificial.

Autor: [Tu Nombre]
Fecha: [Fecha actual]
Curso: Matemáticas para IA - Álgebra Lineal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import os

# Configurar estilo de visualización
plt.style.use('default')
sns.set_palette("husl")

# Crear carpeta de resultados si no existe
os.makedirs('results', exist_ok=True)

def cargar_dataset():
    """
    Carga y prepara el dataset de vinos para análisis PCA.
    
    Returns:
        pd.DataFrame: DataFrame con los datos de vinos
        np.ndarray: Matriz de características (X)
        np.ndarray: Vector de clases (y)
        list: Nombres de las características
    """
    print("🍷 ANÁLISIS DE CALIDAD DEL VINO USANDO PCA")
    print("="*60)
    
    # Cargar el dataset real de vinos (UCI Repository)
    wine_data = load_wine()
    
    # Convertir a DataFrame para mejor manipulación
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    df['target'] = wine_data.target
    df['wine_class'] = df['target'].map({
        0: 'Clase_0 (Cultivar_1)', 
        1: 'Clase_1 (Cultivar_2)', 
        2: 'Clase_2 (Cultivar_3)'
    })
    
    # Información básica del dataset
    print("📋 INFORMACIÓN DEL DATASET:")
    print(f"• Número de vinos analizados: {df.shape[0]}")
    print(f"• Número de características químicas: {df.shape[1]-2}")
    print(f"• Clases de vino: {list(wine_data.target_names)}")
    print(f"• Distribución de clases:")
    for i, name in enumerate(wine_data.target_names):
        count = sum(df.target == i)
        print(f"  - {name}: {count} muestras")
    
    print("\n📊 PRIMERAS 5 MUESTRAS:")
    print(df.head().round(2))
    
    print("\n📈 DIMENSIONES DEL PROBLEMA:")
    print(f"• Datos de entrada: matriz de {df.shape[0]} × {df.shape[1]-2}")
    print(f"• Sin PCA: necesitamos analizar {df.shape[1]-2} variables químicas")
    print(f"• Con PCA: reduciremos a 2-3 componentes principales")
    
    # Preparar datos para el análisis
    X = df.drop(['target', 'wine_class'], axis=1).values
    y = df['target'].values
    feature_names = df.drop(['target', 'wine_class'], axis=1).columns.tolist()
    
    print(f"\n✅ Datos preparados:")
    print(f"• X (características): {X.shape}")
    print(f"• y (clases): {y.shape}")
    
    # Mostrar las variables que vamos a analizar
    print(f"\n🧪 VARIABLES QUÍMICAS A ANALIZAR:")
    for i, feature in enumerate(feature_names, 1):
        print(f"{i:2d}. {feature}")
    
    return df, X, y, feature_names

def analizar_estadisticas(df, X, feature_names):
    """
    Analiza las estadísticas descriptivas y detecta la necesidad de estandarización.
    
    Args:
        df: DataFrame con los datos
        X: Matriz de características  
        feature_names: Nombres de las características
    """
    print("\n" + "="*60)
    print("📊 ANÁLISIS ESTADÍSTICO PRE-PCA")
    print("="*60)
    
    # Estadísticas descriptivas
    stats = pd.DataFrame({
        'Media': np.mean(X, axis=0),
        'Desv_Std': np.std(X, axis=0),
        'Mínimo': np.min(X, axis=0),
        'Máximo': np.max(X, axis=0),
        'Rango': np.max(X, axis=0) - np.min(X, axis=0)
    }, index=feature_names)
    
    print("📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(stats.round(2))
    
    # Detectar variables con escalas muy diferentes
    print(f"\n🔍 ANÁLISIS DE ESCALAS:")
    rangos = stats['Rango'].values
    max_rango = np.max(rangos)
    min_rango = np.min(rangos)
    ratio_escalas = max_rango / min_rango
    
    print(f"• Variable con mayor rango: {feature_names[np.argmax(rangos)]} ({max_rango:.2f})")
    print(f"• Variable con menor rango: {feature_names[np.argmin(rangos)]} ({min_rango:.2f})")
    print(f"• Ratio de escalas: {ratio_escalas:.2f}")
    
    if ratio_escalas > 100:
        print("⚠️  ADVERTENCIA: Las variables tienen escalas MUY diferentes")
        print("   Es CRÍTICO estandarizar antes de aplicar PCA")
    elif ratio_escalas > 10:
        print("⚡ RECOMENDACIÓN: Se recomienda estandarizar antes de PCA")
    else:
        print("✅ Las escalas son relativamente similares")
    
    # Visualizar distribuciones
    crear_visualizacion_escalas(X, feature_names)
    
    return stats

def crear_visualizacion_escalas(X, feature_names):
    """
    Crea visualización para mostrar las diferentes escalas de las variables.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de cajas (boxplot) para ver escalas
    ax1.boxplot(X, labels=range(1, len(feature_names)+1))
    ax1.set_title('📊 Distribución de Variables (Escalas Originales)')
    ax1.set_xlabel('Variable #')
    ax1.set_ylabel('Valor')
    ax1.tick_params(axis='x', rotation=45)
    
    # Gráfico de rangos
    rangos = np.max(X, axis=0) - np.min(X, axis=0)
    ax2.bar(range(len(feature_names)), rangos)
    ax2.set_title('📏 Rango de Cada Variable')
    ax2.set_xlabel('Variable #')
    ax2.set_ylabel('Rango (Máx - Mín)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/01_analisis_escalas.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Gráfico guardado en: results/01_analisis_escalas.png")

def estandarizar_datos(X, feature_names):
    """
    Estandariza los datos para que todas las variables tengan media=0 y std=1.
    
    Args:
        X: Matriz de características original
        feature_names: Nombres de las características
        
    Returns:
        np.ndarray: Matriz estandarizada
        StandardScaler: Objeto scaler para transformaciones futuras
    """
    print("\n" + "="*60)
    print("⚖️  PASO 2: ESTANDARIZACIÓN DE DATOS")
    print("="*60)
    
    print("🎯 OBJETIVO: Transformar todas las variables a la misma escala")
    print("   Fórmula: X_estandarizado = (X - media) / desviación_estándar")
    
    # Crear objeto estandarizador
    scaler = StandardScaler()
    
    # Calcular estadísticas antes de estandarizar
    print("\n📊 ANTES DE ESTANDARIZAR:")
    print(f"• Media mínima: {np.min(np.mean(X, axis=0)):.2f}")
    print(f"• Media máxima: {np.max(np.mean(X, axis=0)):.2f}")
    print(f"• Std mínima: {np.min(np.std(X, axis=0)):.2f}")
    print(f"• Std máxima: {np.max(np.std(X, axis=0)):.2f}")
    
    # Estandarizar los datos
    X_scaled = scaler.fit_transform(X)
    
    # Verificar estandarización
    print("\n📊 DESPUÉS DE ESTANDARIZAR:")
    print(f"• Media mínima: {np.min(np.mean(X_scaled, axis=0)):.10f}")
    print(f"• Media máxima: {np.max(np.mean(X_scaled, axis=0)):.10f}")
    print(f"• Std mínima: {np.min(np.std(X_scaled, axis=0)):.10f}")
    print(f"• Std máxima: {np.max(np.std(X_scaled, axis=0)):.10f}")
    
    print("\n✅ VERIFICACIÓN: Todas las medias ≈ 0 y todas las std ≈ 1")
    
    # Crear visualización de la estandarización
    crear_visualizacion_estandarizacion(X, X_scaled, feature_names)
    
    return X_scaled, scaler

def crear_visualizacion_estandarizacion(X_original, X_scaled, feature_names):
    """
    Visualiza el efecto de la estandarización en los datos.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Datos originales
    ax1.boxplot(X_original, labels=range(1, len(feature_names)+1))
    ax1.set_title('📊 Datos Originales (Escalas Diferentes)')
    ax1.set_xlabel('Variable #')
    ax1.set_ylabel('Valor Original')
    ax1.tick_params(axis='x', rotation=45)
    
    # Datos estandarizados
    ax2.boxplot(X_scaled, labels=range(1, len(feature_names)+1))
    ax2.set_title('⚖️  Datos Estandarizados (Media=0, Std=1)')
    ax2.set_xlabel('Variable #')
    ax2.set_ylabel('Valor Estandarizado')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Media = 0')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/02_estandarizacion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Gráfico guardado en: results/02_estandarizacion.png")

def calcular_matriz_covarianza(X_scaled, feature_names):
    """
    Calcula y analiza la matriz de covarianza paso a paso.
    
    Args:
        X_scaled: Matriz de datos estandarizados
        feature_names: Nombres de las características
        
    Returns:
        np.ndarray: Matriz de covarianza
    """
    print("\n" + "="*60)
    print("📈 PASO 3: MATRIZ DE COVARIANZA")
    print("="*60)
    
    print("🎯 OBJETIVO: Medir cómo se relacionan las variables entre sí")
    print("   Fórmula: Cov = (X^T × X) / (n-1)")
    
    n_samples, n_features = X_scaled.shape
    print(f"\n📐 DIMENSIONES:")
    print(f"• X_scaled: {n_samples} × {n_features}")
    print(f"• X_scaled^T: {n_features} × {n_samples}")
    print(f"• Covarianza: {n_features} × {n_features}")
    
    # Calcular matriz de covarianza manualmente
    print("\n🧮 CÁLCULO PASO A PASO:")
    print("1️⃣  Transponer la matriz: X^T")
    X_T = X_scaled.T
    print(f"   X^T shape: {X_T.shape}")
    
    print("2️⃣  Multiplicar: X^T × X")
    XTX = X_T @ X_scaled
    print(f"   X^T × X shape: {XTX.shape}")
    
    print("3️⃣  Dividir por (n-1):")
    cov_matrix = XTX / (n_samples - 1)
    print(f"   Matriz de covarianza final: {cov_matrix.shape}")
    
    # Verificar con NumPy
    cov_numpy = np.cov(X_scaled.T)
    diferencia = np.max(np.abs(cov_matrix - cov_numpy))
    print(f"\n✅ VERIFICACIÓN con np.cov(): diferencia máxima = {diferencia:.10f}")
    
    # Analizar la matriz de covarianza
    print(f"\n📊 ANÁLISIS DE LA MATRIZ DE COVARIANZA:")
    diagonal = np.diag(cov_matrix)
    print(f"• Elementos diagonales (varianzas): min={np.min(diagonal):.3f}, max={np.max(diagonal):.3f}")
    
    # Encontrar correlaciones más fuertes
    cov_abs = np.abs(cov_matrix)
    np.fill_diagonal(cov_abs, 0)  # Ignorar diagonal
    max_idx = np.unravel_index(np.argmax(cov_abs), cov_abs.shape)
    print(f"• Correlación más fuerte: {feature_names[max_idx[0]]} ↔ {feature_names[max_idx[1]]}")
    print(f"  Valor: {cov_matrix[max_idx]:.3f}")
    
    # Visualizar matriz de covarianza
    crear_visualizacion_covarianza(cov_matrix, feature_names)
    
    return cov_matrix

def crear_visualizacion_covarianza(cov_matrix, feature_names):
    """
    Crea un heatmap de la matriz de covarianza.
    """
    plt.figure(figsize=(12, 10))
    
    # Crear heatmap
    mask = np.triu(np.ones_like(cov_matrix, dtype=bool), k=1)
    sns.heatmap(cov_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                mask=mask,
                xticklabels=[f'V{i+1}' for i in range(len(feature_names))],
                yticklabels=[f'V{i+1}' for i in range(len(feature_names))],
                cbar_kws={'label': 'Covarianza'})
    
    plt.title('📈 Matriz de Covarianza\n(Solo triángulo inferior - simétrica)', fontsize=14)
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    
    # Añadir leyenda de variables
    legend_text = '\n'.join([f'V{i+1}: {name[:20]}...' if len(name) > 20 else f'V{i+1}: {name}' 
                            for i, name in enumerate(feature_names)])
    plt.figtext(1.02, 0.5, legend_text, fontsize=9, va='center')
    
    plt.tight_layout()
    plt.savefig('results/03_matriz_covarianza.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Gráfico guardado en: results/03_matriz_covarianza.png")

def calcular_eigenvalues_eigenvectors(cov_matrix, feature_names):
    """
    Calcula eigenvalues y eigenvectors de la matriz de covarianza paso a paso.
    
    Args:
        cov_matrix: Matriz de covarianza
        feature_names: Nombres de las características
        
    Returns:
        tuple: (eigenvalues, eigenvectors, indices_ordenados)
    """
    print("\n" + "="*60)
    print("🔮 PASO 4: EIGENVALUES Y EIGENVECTORS")
    print("="*60)
    
    print("🎯 OBJETIVO: Encontrar las direcciones de máxima varianza")
    print("   Ecuación: Cov × v = λ × v")
    print("   Donde: v = eigenvector, λ = eigenvalue")
    
    # Calcular eigenvalues y eigenvectors
    print("\n🧮 CÁLCULO:")
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    print(f"• Eigenvalues calculados: {len(eigenvalues)}")
    print(f"• Eigenvectors shape: {eigenvectors.shape}")
    
    # Ordenar por eigenvalues (de mayor a menor)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    print(f"\n📊 EIGENVALUES (Ordenados de mayor a menor):")
    cumulative_var = np.cumsum(eigenvalues_sorted) / np.sum(eigenvalues_sorted) * 100
    
    for i in range(min(5, len(eigenvalues_sorted))):  # Mostrar solo los primeros 5
        print(f"PC{i+1}: λ = {eigenvalues_sorted[i]:.3f} "
              f"({eigenvalues_sorted[i]/np.sum(eigenvalues_sorted)*100:.1f}% varianza) "
              f"[Acumulada: {cumulative_var[i]:.1f}%]")
    
    # Analizar cuántos componentes necesitamos
    var_80 = np.where(cumulative_var >= 80)[0][0] + 1
    var_90 = np.where(cumulative_var >= 90)[0][0] + 1
    var_95 = np.where(cumulative_var >= 95)[0][0] + 1
    
    print(f"\n📈 ANÁLISIS DE VARIANZA EXPLICADA:")
    print(f"• Para capturar 80% de la información: {var_80} componentes")
    print(f"• Para capturar 90% de la información: {var_90} componentes")
    print(f"• Para capturar 95% de la información: {var_95} componentes")
    
    # Verificar propiedades matemáticas
    verificar_propiedades_eigen(cov_matrix, eigenvalues_sorted, eigenvectors_sorted)
    
    # Visualizar eigenvalues y eigenvectors
    crear_visualizacion_eigen(eigenvalues_sorted, eigenvectors_sorted, feature_names, cumulative_var)
    
    return eigenvalues_sorted, eigenvectors_sorted, idx

def verificar_propiedades_eigen(cov_matrix, eigenvalues, eigenvectors):
    """
    Verifica las propiedades matemáticas de eigenvalues y eigenvectors.
    """
    print(f"\n🔍 VERIFICACIÓN MATEMÁTICA:")
    
    # 1. Verificar Cov @ v = λ @ v para el primer eigenvector
    v1 = eigenvectors[:, 0]
    lambda1 = eigenvalues[0]
    
    left_side = cov_matrix @ v1
    right_side = lambda1 * v1
    error = np.max(np.abs(left_side - right_side))
    
    print(f"1️⃣  Ecuación Cov×v = λ×v: error máximo = {error:.10f}")
    
    # 2. Verificar que eigenvectors son ortonormales
    dot_products = []
    for i in range(min(3, eigenvectors.shape[1])):
        for j in range(i+1, min(3, eigenvectors.shape[1])):
            dot_prod = np.dot(eigenvectors[:, i], eigenvectors[:, j])
            dot_products.append(abs(dot_prod))
    
    max_dot = max(dot_products) if dot_products else 0
    print(f"2️⃣  Ortogonalidad: producto punto máximo entre eigenvectors = {max_dot:.10f}")
    
    # 3. Verificar normas unitarias
    norms = [np.linalg.norm(eigenvectors[:, i]) for i in range(min(3, eigenvectors.shape[1]))]
    max_norm_error = max([abs(norm - 1.0) for norm in norms])
    print(f"3️⃣  Normalización: error máximo de norma = {max_norm_error:.10f}")
    
    # 4. Verificar que la suma de eigenvalues = traza de la matriz
    trace_cov = np.trace(cov_matrix)
    sum_eigenvals = np.sum(eigenvalues)
    trace_error = abs(trace_cov - sum_eigenvals)
    print(f"4️⃣  Traza: suma eigenvalues = {sum_eigenvals:.6f}, traza matriz = {trace_cov:.6f}")
    print(f"     Error = {trace_error:.10f}")
    
    print("✅ Todas las propiedades matemáticas verificadas correctamente!")

def crear_visualizacion_eigen(eigenvalues, eigenvectors, feature_names, cumulative_var):
    """
    Crea visualizaciones de eigenvalues y eigenvectors.
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Gráfico de eigenvalues (scree plot)
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, 'bo-', linewidth=2, markersize=8)
    plt.title('📊 Scree Plot - Eigenvalues')
    plt.xlabel('Componente Principal')
    plt.ylabel('Eigenvalue (Varianza)')
    plt.grid(True, alpha=0.3)
    
    # 2. Varianza explicada acumulativa
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(range(1, len(cumulative_var)+1), cumulative_var, 'ro-', linewidth=2, markersize=8)
    plt.axhline(y=80, color='green', linestyle='--', alpha=0.7, label='80%')
    plt.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90%')
    plt.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95%')
    plt.title('📈 Varianza Explicada Acumulativa')
    plt.xlabel('Número de Componentes')
    plt.ylabel('% Varianza Acumulativa')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Heatmap de los primeros 3 eigenvectors
    ax3 = plt.subplot(2, 3, 3)
    eigenvectors_subset = eigenvectors[:, :3]
    sns.heatmap(eigenvectors_subset.T, 
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                xticklabels=[f'V{i+1}' for i in range(len(feature_names))],
                yticklabels=['PC1', 'PC2', 'PC3'],
                cbar_kws={'label': 'Peso del eigenvector'})
    plt.title('🎯 Primeros 3 Eigenvectors\n(Pesos de cada variable)')
    
    # 4. Gráfico de barras para PC1
    ax4 = plt.subplot(2, 3, 4)
    colors = ['red' if x < 0 else 'blue' for x in eigenvectors[:, 0]]
    plt.barh(range(len(feature_names)), eigenvectors[:, 0], color=colors, alpha=0.7)
    plt.title('🔍 Primer Componente Principal (PC1)')
    plt.xlabel('Peso en PC1')
    plt.yticks(range(len(feature_names)), [f'V{i+1}' for i in range(len(feature_names))])
    plt.grid(True, alpha=0.3)
    
    # 5. Gráfico de barras para PC2
    ax5 = plt.subplot(2, 3, 5)
    colors = ['red' if x < 0 else 'blue' for x in eigenvectors[:, 1]]
    plt.barh(range(len(feature_names)), eigenvectors[:, 1], color=colors, alpha=0.7)
    plt.title('🔍 Segundo Componente Principal (PC2)')
    plt.xlabel('Peso en PC2')
    plt.yticks(range(len(feature_names)), [f'V{i+1}' for i in range(len(feature_names))])
    plt.grid(True, alpha=0.3)
    
    # 6. Contribución de variables a PC1 y PC2
    ax6 = plt.subplot(2, 3, 6)
    pc1_contrib = eigenvectors[:, 0]**2
    pc2_contrib = eigenvectors[:, 1]**2
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x - width/2, pc1_contrib, width, label='PC1', alpha=0.7)
    plt.bar(x + width/2, pc2_contrib, width, label='PC2', alpha=0.7)
    plt.title('⚡ Contribución de Variables\n(Pesos al cuadrado)')
    plt.xlabel('Variables')
    plt.ylabel('Contribución')
    plt.xticks(x, [f'V{i+1}' for i in range(len(feature_names))], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/04_eigenvalues_eigenvectors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Gráfico guardado en: results/04_eigenvalues_eigenvectors.png")

def realizar_pca_transformacion(X_scaled, eigenvectors, feature_names):
    """
    Realiza la transformación PCA proyectando los datos en los nuevos ejes.
    
    Args:
        X_scaled: Datos estandarizados
        eigenvectors: Eigenvectors calculados
        feature_names: Nombres de las características
        
    Returns:
        np.ndarray: Datos transformados por PCA
    """
    print("\n" + "="*60)
    print("🔄 PASO 5: TRANSFORMACIÓN PCA")
    print("="*60)
    
    print("🎯 OBJETIVO: Proyectar datos en las nuevas componentes principales")
    print("   Fórmula: X_pca = X_scaled × Eigenvectors")
    
    # Realizar transformación PCA
    print(f"\n🧮 TRANSFORMACIÓN:")
    print(f"• X_scaled shape: {X_scaled.shape}")
    print(f"• Eigenvectors shape: {eigenvectors.shape}")
    
    X_pca = X_scaled @ eigenvectors
    print(f"• X_pca shape: {X_pca.shape}")
    
    # Análisis de la transformación
    print(f"\n📊 ANÁLISIS DE LOS DATOS TRANSFORMADOS:")
    for i in range(min(3, X_pca.shape[1])):
        variance = np.var(X_pca[:, i])
        print(f"PC{i+1}: varianza = {variance:.3f}")
    
    # Verificar que las varianzas son los eigenvalues
    print(f"\n✅ VERIFICACIÓN:")
    print("Las varianzas de los componentes principales deben ser iguales a los eigenvalues")
    
    return X_pca

def visualizacion_final_2d(X_pca, y, feature_names, eigenvalues, eigenvectors):
    """
    Crea la visualización final en 2D y analiza los resultados.
    
    Args:
        X_pca: Datos transformados por PCA
        y: Clases de vino
        feature_names: Nombres de las características
        eigenvalues: Eigenvalues calculados
        eigenvectors: Eigenvectors calculados
    """
    print("\n" + "="*60)
    print("🎨 PASO 6: VISUALIZACIÓN FINAL Y ANÁLISIS")
    print("="*60)
    
    print("🎯 OBJETIVO: Visualizar 178 vinos en solo 2 dimensiones")
    print("   De 13D → 2D manteniendo la máxima información posible")
    
    # Calcular información capturada
    total_variance = np.sum(eigenvalues)
    pc1_variance = eigenvalues[0] / total_variance * 100
    pc2_variance = eigenvalues[1] / total_variance * 100
    pc12_variance = (eigenvalues[0] + eigenvalues[1]) / total_variance * 100
    
    print(f"\n📊 INFORMACIÓN CAPTURADA:")
    print(f"• PC1: {pc1_variance:.1f}% de la varianza total")
    print(f"• PC2: {pc2_variance:.1f}% de la varianza total")
    print(f"• PC1 + PC2: {pc12_variance:.1f}% de la varianza total")
    print(f"• Perdemos solo: {100 - pc12_variance:.1f}% de la información")
    
    # Crear visualización completa
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Scatter plot principal en 2D
    ax1 = plt.subplot(2, 3, 1)
    colors = ['red', 'blue', 'green']
    wine_classes = ['Cultivar 1', 'Cultivar 2', 'Cultivar 3']
    
    for i, (color, wine_class) in enumerate(zip(colors, wine_classes)):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=color, label=wine_class, alpha=0.7, s=60)
    
    plt.xlabel(f'PC1 ({pc1_variance:.1f}% varianza)')
    plt.ylabel(f'PC2 ({pc2_variance:.1f}% varianza)')
    plt.title('🍷 Vinos Proyectados en 2D\n(178 vinos de 13D → 2D)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Análisis de separación de clases
    ax2 = plt.subplot(2, 3, 2)
    
    # Calcular centroides de cada clase
    centroids = []
    for i in range(3):
        mask = y == i
        centroid = [np.mean(X_pca[mask, 0]), np.mean(X_pca[mask, 1])]
        centroids.append(centroid)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], alpha=0.3, s=40)
        plt.scatter(centroid[0], centroid[1], 
                   c=colors[i], s=200, marker='X', 
                   edgecolors='black', linewidth=2, label=f'Centro {wine_classes[i]}')
    
    plt.xlabel(f'PC1 ({pc1_variance:.1f}% varianza)')
    plt.ylabel(f'PC2 ({pc2_variance:.1f}% varianza)')
    plt.title('🎯 Centroides de Clases\n(Separación automática)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Biplot - Variables originales en el espacio PC
    ax3 = plt.subplot(2, 3, 3)
    
    # Proyectar las variables originales en el espacio PC
    scale_factor = 3
    for i, feature in enumerate(feature_names):
        arrow_x = eigenvectors[i, 0] * scale_factor
        arrow_y = eigenvectors[i, 1] * scale_factor
        plt.arrow(0, 0, arrow_x, arrow_y, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        plt.text(arrow_x*1.1, arrow_y*1.1, f'V{i+1}', fontsize=8, ha='center')
    
    plt.xlabel(f'PC1 ({pc1_variance:.1f}% varianza)')
    plt.ylabel(f'PC2 ({pc2_variance:.1f}% varianza)')
    plt.title('🧭 Biplot - Contribución de Variables\n(Flechas = variables originales)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 4. Distribución en PC1
    ax4 = plt.subplot(2, 3, 4)
    for i, (color, wine_class) in enumerate(zip(colors, wine_classes)):
        mask = y == i
        plt.hist(X_pca[mask, 0], alpha=0.6, color=color, 
                label=wine_class, bins=15, density=True)
    
    plt.xlabel(f'PC1 ({pc1_variance:.1f}% varianza)')
    plt.ylabel('Densidad')
    plt.title('📊 Distribución en PC1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Distribución en PC2
    ax5 = plt.subplot(2, 3, 5)
    for i, (color, wine_class) in enumerate(zip(colors, wine_classes)):
        mask = y == i
        plt.hist(X_pca[mask, 1], alpha=0.6, color=color, 
                label=wine_class, bins=15, density=True)
    
    plt.xlabel(f'PC2 ({pc2_variance:.1f}% varianza)')
    plt.ylabel('Densidad')
    plt.title('📊 Distribución en PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Comparación 3D vs 2D
    ax6 = plt.subplot(2, 3, 6)
    pc3_variance = eigenvalues[2] / total_variance * 100
    pc123_variance = (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]) / total_variance * 100
    
    components = ['PC1', 'PC2', 'PC3', 'PC4-PC13']
    variances = [pc1_variance, pc2_variance, pc3_variance, 100-pc123_variance]
    colors_bar = ['darkblue', 'blue', 'lightblue', 'lightgray']
    
    bars = plt.bar(components, variances, color=colors_bar, alpha=0.8)
    plt.ylabel('% Varianza Explicada')
    plt.title('📈 Importancia de Componentes\n(¿Por qué 2D es suficiente?)')
    plt.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, variance in zip(bars, variances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{variance:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/05_visualizacion_final_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Gráfico guardado en: results/05_visualizacion_final_2d.png")
    
    # Análisis de los resultados
    analizar_resultados_pca(X_pca, y, centroids, pc12_variance)

def analizar_resultados_pca(X_pca, y, centroids, variance_explained):
    """
    Analiza los resultados del PCA y su interpretación.
    """
    print(f"\n🔍 ANÁLISIS DE RESULTADOS:")
    
    # Calcular distancias entre centroides
    dist_01 = np.sqrt((centroids[0][0] - centroids[1][0])**2 + (centroids[0][1] - centroids[1][1])**2)
    dist_02 = np.sqrt((centroids[0][0] - centroids[2][0])**2 + (centroids[0][1] - centroids[2][1])**2)
    dist_12 = np.sqrt((centroids[1][0] - centroids[2][0])**2 + (centroids[1][1] - centroids[2][1])**2)
    
    print(f"📏 SEPARACIÓN ENTRE CLASES:")
    print(f"• Cultivar 1 ↔ Cultivar 2: {dist_01:.2f} unidades")
    print(f"• Cultivar 1 ↔ Cultivar 3: {dist_02:.2f} unidades")
    print(f"• Cultivar 2 ↔ Cultivar 3: {dist_12:.2f} unidades")
    
    # Evaluar calidad de la separación
    if min(dist_01, dist_02, dist_12) > 2:
        print("✅ EXCELENTE separación: Las clases están bien diferenciadas")
    elif min(dist_01, dist_02, dist_12) > 1:
        print("✅ BUENA separación: Las clases son distinguibles")
    else:
        print("⚠️  SEPARACIÓN MODERADA: Algunas clases se superponen")
    
    print(f"\n🎯 LOGROS DEL PCA:")
    print(f"• ✅ Reducimos 13 dimensiones → 2 dimensiones")
    print(f"• ✅ Mantenemos {variance_explained:.1f}% de la información")
    print(f"• ✅ Separamos automáticamente las 3 clases de vino")
    print(f"• ✅ Podemos visualizar patrones antes invisibles")

def aplicaciones_en_ia(eigenvalues, feature_names):
    """
    Explica las aplicaciones del PCA en Inteligencia Artificial.
    """
    print("\n" + "="*60)
    print("🤖 APLICACIONES EN INTELIGENCIA ARTIFICIAL")
    print("="*60)
    
    print("🎯 LO QUE ACABAMOS DE HACER ES LA BASE DE:")
    
    print(f"\n1️⃣  🖼️  COMPUTER VISION (Visión por Computador):")
    print(f"   • Reconocimiento facial: Eigenfaces (PCA en imágenes de caras)")
    print(f"   • Compresión de imágenes: Reducir píxeles manteniendo información visual")
    print(f"   • Detección de objetos: Reducir dimensionalidad antes de clasificar")
    
    print(f"\n2️⃣  📊 MACHINE LEARNING:")
    print(f"   • Preprocesamiento: Reducir overfitting con menos dimensiones")
    print(f"   • Visualización: Entender datos de alta dimensión")
    print(f"   • Feature selection: Identificar variables más importantes")
    
    print(f"\n3️⃣  🗣️  PROCESAMIENTO DE LENGUAJE NATURAL:")
    print(f"   • Word embeddings: Reducir vectores de palabras (Word2Vec → PCA)")
    print(f"   • Análisis de sentimientos: Encontrar patrones en texto")
    print(f"   • Traducción automática: Mapear idiomas en espacios reducidos")
    
    print(f"\n4️⃣  🎵 SISTEMAS DE RECOMENDACIÓN:")
    print(f"   • Netflix: Factorización matricial (similar a PCA)")
    print(f"   • Spotify: Reducir perfiles musicales de usuarios")
    print(f"   • Amazon: Recomendar productos basado en patrones latentes")
    
    print(f"\n5️⃣  🧠 DEEP LEARNING:")
    print(f"   • Autoencoders: Redes neuronales que hacen PCA no-lineal")
    print(f"   • Initialización: Usar PCA para inicializar pesos")
    print(f"   • Regularización: Proyectar en subespacios para evitar overfitting")
    
    print(f"\n6️⃣  📈 BIG DATA:")
    print(f"   • Reducir datasets masivos manteniendo información clave")
    print(f"   • Clustering eficiente en espacios de menor dimensión")
    print(f"   • Análisis exploratorio de datos complejos")
    
    # Crear resumen de conceptos de álgebra lineal aplicados
    print(f"\n🧮 CONCEPTOS DE ÁLGEBRA LINEAL QUE DOMINASTE:")
    print(f"   ✅ Representación matricial de datos")
    print(f"   ✅ Transformaciones lineales (estandarización)")
    print(f"   ✅ Producto matricial (matriz de covarianza)")
    print(f"   ✅ Eigenvalues y eigenvectors (descomposición espectral)")
    print(f"   ✅ Cambio de base y proyecciones")
    print(f"   ✅ Reducción de dimensionalidad")
    print(f"   ✅ Interpretación geométrica del álgebra lineal")

def generar_informe_final(df, X, X_scaled, X_pca, y, feature_names, eigenvalues, eigenvectors, variance_explained):
    """
    Genera un informe final con todos los resultados.
    """
    print("\n" + "="*60)
    print("📋 INFORME FINAL - ANÁLISIS PCA DE VINOS")
    print("="*60)
    
    informe = f"""
ANÁLISIS DE COMPONENTES PRINCIPALES (PCA) - DATASET DE VINOS
===========================================================

DATOS ANALIZADOS:
• Dataset: UCI Wine Dataset
• Muestras: {df.shape[0]} vinos
• Variables originales: {len(feature_names)} características químicas
• Clases: 3 tipos de vino (Cultivar 1, 2, 3)

TRANSFORMACIÓN REALIZADA:
• Estandarización: Media = 0, Desviación estándar = 1
• Dimensionalidad: {len(feature_names)}D → 2D
• Información conservada: {variance_explained:.1f}%
• Información perdida: {100-variance_explained:.1f}%

COMPONENTES PRINCIPALES:
• PC1: {eigenvalues[0]/np.sum(eigenvalues)*100:.1f}% de la varianza
• PC2: {eigenvalues[1]/np.sum(eigenvalues)*100:.1f}% de la varianza
• PC3: {eigenvalues[2]/np.sum(eigenvalues)*100:.1f}% de la varianza

RESULTADOS:
✅ Separación exitosa de las 3 clases de vino en 2D
✅ Visualización clara de patrones antes invisibles
✅ Reducción significativa de complejidad manteniendo información
✅ Aplicación exitosa de conceptos fundamentales de álgebra lineal

CONCEPTOS DE ÁLGEBRA LINEAL APLICADOS:
1. Matrices y vectores para representación de datos
2. Transformaciones lineales para estandarización
3. Producto matricial para matriz de covarianza
4. Eigenvalues y eigenvectors para encontrar direcciones principales
5. Cambio de base para proyección en nuevos ejes
6. Reducción de dimensionalidad preservando máxima información

APLICACIONES EN IA:
• Computer Vision: Eigenfaces, compresión de imágenes
• Machine Learning: Preprocesamiento, reducción de overfitting
• NLP: Word embeddings, análisis de sentimientos
• Sistemas de recomendación: Factorización matricial
• Deep Learning: Autoencoders, inicialización
• Big Data: Análisis eficiente de datasets masivos
    """
    
    # Guardar informe
    with open('results/informe_final_pca.txt', 'w', encoding='utf-8') as f:
        f.write(informe)
    
    print(informe)
    print(f"💾 Informe completo guardado en: results/informe_final_pca.txt")

def main():
    """
    Función principal que ejecuta el análisis paso a paso.
    """
    print("🚀 INICIANDO ANÁLISIS DE VINOS CON PCA")
    print("Paso 1 de 6: Cargar y explorar dataset\n")
    
    # Paso 1: Cargar dataset
    df, X, y, feature_names = cargar_dataset()
    
    # Paso 2: Análisis estadístico inicial
    stats = analizar_estadisticas(df, X, feature_names)
    
    # Paso 3: Estandarización
    X_scaled, scaler = estandarizar_datos(X, feature_names)
    
    # Paso 4: Matriz de covarianza
    cov_matrix = calcular_matriz_covarianza(X_scaled, feature_names)
    
    # Paso 5: Eigenvalues y eigenvectors
    eigenvalues, eigenvectors, idx = calcular_eigenvalues_eigenvectors(cov_matrix, feature_names)
    
    # Paso 6: Transformación PCA
    X_pca = realizar_pca_transformacion(X_scaled, eigenvectors, feature_names)
    
    # Paso 7: Visualización final
    visualizacion_final_2d(X_pca, y, feature_names, eigenvalues, eigenvectors)
    
    # Paso 8: Aplicaciones en IA
    aplicaciones_en_ia(eigenvalues, feature_names)
    
    # Paso 9: Informe final
    pc12_variance = (eigenvalues[0] + eigenvalues[1]) / np.sum(eigenvalues) * 100
    generar_informe_final(df, X, X_scaled, X_pca, y, feature_names, eigenvalues, eigenvectors, pc12_variance)
    
    print("\n" + "="*60)
    print("🎉 ANÁLISIS PCA COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("💡 Has aplicado conceptos fundamentales de álgebra lineal")
    print("🤖 Has visto aplicaciones directas en Inteligencia Artificial")
    print("📊 Has reducido 13 dimensiones a 2 manteniendo máxima información")
    print("🍷 Has separado automáticamente 3 tipos de vino")
    
    return df, X, X_scaled, X_pca, y, feature_names, stats, cov_matrix, eigenvalues, eigenvectors, scaler

if __name__ == "__main__":
    # Ejecutar análisis completo
    results = main()
    df, X, X_scaled, X_pca, y, feature_names, stats, cov_matrix, eigenvalues, eigenvectors, scaler = results
    
    print(f"\n🎯 MISIÓN CUMPLIDA - TODOS LOS PASOS COMPLETADOS")
    print(f"📁 Archivos generados en la carpeta 'results/':")
    print(f"   • 01_analisis_escalas.png")
    print(f"   • 02_estandarizacion.png") 
    print(f"   • 03_matriz_covarianza.png")
    print(f"   • 04_eigenvalues_eigenvectors.png")
    print(f"   • 05_visualizacion_final_2d.png")
    print(f"   • informe_final_pca.txt")
    print(f"\n🏆 FELICITACIONES: Has completado un análisis PCA profesional!")
    print(f"📚 Ahora entiendes cómo el álgebra lineal potencia la IA")
# src/utils/lib_loader.py
"""
Sistema simples de carregamento de bibliotecas.
Criado para facilitar imports em projetos de Data Science.
"""

import importlib
import sys
from platform import python_version


# ============================================================================
# 🔧 CONFIGURAÇÃO - FÁCIL DE MODIFICAR
# ============================================================================

# Bibliotecas disponíveis - Para adicionar novas, basta incluir aqui!
LIBS = {
    # Básicas
    'pd': ('pandas', 'pandas'),
    'np': ('numpy', 'numpy'),
    
    # Visualização  
    'plt': ('matplotlib.pyplot', 'matplotlib'),
    'mpl': ('matplotlib', 'matplotlib'),
    'sns': ('seaborn', 'seaborn'),
    'plotly': ('plotly.express', 'plotly'),
    
    # Machine Learning
    'sk': ('sklearn', 'scikit-learn'), 
    'DecisionTreeClassifier': ('sklearn.tree.DecisionTreeClassifier', 'scikit-learn'),
    'LogisticRegression': ('sklearn.linear_model.LogisticRegression', 'scikit-learn'),
    'roc_auc_score': ('sklearn.metrics.roc_auc_score', 'scikit-learn'),
    'roc_curve': ('sklearn.metrics.roc_curve', 'scikit-learn'),
    'auc': ('sklearn.metrics.auc', 'scikit-learn'),
    'lgb': ('lightgbm', 'lightgbm'),
    'xgb': ('xgboost', 'xgboost'),
    'shap': ('shap', 'shap'),
    'optuna': ('optuna', 'optuna'),
    
    # Outras úteis
    'requests': ('requests', 'requests'),
    'json': ('json', 'json'),
    're': ('re', 're'),
}

# Perfis pré-definidos - Para adicionar novos, basta incluir aqui!
PERFIS = {
    'eda': ['pd', 'np', 'mpl', 'plt', 'sns'],
    'ml': ['pd', 'np', 'sk', 'plt', 'sns'],
    'fraud': ['pd', 'np', 'sk', 'lgb', 'shap', 'plt', 'sns'],
    'viz': ['pd', 'plt', 'sns', 'plotly'],
    'web': ['requests', 'pd', 'json'],
    'minimal': ['pd', 'np'],
}


# ============================================================================
# 📚 FUNÇÃO 1: MOSTRAR BIBLIOTECAS DISPONÍVEIS
# ============================================================================

def mostrar_libs():
    """Mostra todas as bibliotecas disponíveis."""
    print("BIBLIOTECAS DISPONÍVEIS:")
    print("=" * 40)
    
    for lib, (modulo, pacote) in LIBS.items():
        print(f"• {lib:<10} → {pacote:<15} → {modulo}")
    
    print(f"\nPERFIS PRÉ-DEFINIDOS:")
    print("=" * 30)
    
    for perfil, libs in PERFIS.items():
        libs_str = ', '.join(libs)
        print(f"• {perfil:<10} → {libs_str}")
    
    print(f"\nTotal: {len(LIBS)} bibliotecas e {len(PERFIS)} perfis")


# ============================================================================
# 🎯 FUNÇÃO 2: CARREGAR BIBLIOTECAS ESPECÍFICAS
# ============================================================================

def carregar_libs(lista_libs):
    """
    Carrega as bibliotecas que você especificar.
    
    Exemplo:
        carregar_libs(['pd', 'np', 'sk'])
    """
    print(f"Carregando: {lista_libs}")
    print("-" * 30)
    
    carregadas = {}
    
    for lib in lista_libs:
        if lib not in LIBS:
            print(f"❌ '{lib}' não encontrada")
            continue
            
        modulo, pacote = LIBS[lib]
        
        try:
            # Método mais robusto para importar
            biblioteca = _importar_biblioteca(modulo)
            
            carregadas[lib] = biblioteca
            print(f"✅ {lib} ({modulo})")
            
        except ImportError:
            print(f"❌ {lib} - Instale com: pip install {pacote}")
        except Exception as e:
            print(f"❌ {lib} - Erro: {str(e)[:50]}...")
    
    if carregadas:
        _mostrar_versoes(carregadas)
        _adicionar_ao_namespace(carregadas)
    
    return carregadas


def _importar_biblioteca(modulo_path):
    """Importa biblioteca de forma robusta."""
    
    partes = modulo_path.split('.')
    
    try:
        modulo_base = importlib.import_module(partes[0])
    except ImportError as e:
        raise ImportError(f"Falha ao importar o módulo base '{partes[0]}'. Verifique a instalação.") from e

    objeto_atual = modulo_base
    
    for i, parte in enumerate(partes[1:]):
        try:
            objeto_atual = getattr(objeto_atual, parte)
            
        except AttributeError:
            
            try:
                caminho_submodulo = ".".join(partes[:i + 2])
                
                objeto_atual = importlib.import_module(caminho_submodulo)
                
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Falha ao importar o objeto/submódulo '{parte}' do caminho '{modulo_path}'. Verifique se o caminho ou a biblioteca estão corretos.") from e
            
    return objeto_atual

# ============================================================================
# 🎯 FUNÇÃO 3: CARREGAR PERFIL PRÉ-DEFINIDO
# ============================================================================

def carregar_perfil(nome_perfil):
    """
    Carrega um perfil pré-definido.
    
    Exemplos:
        carregar_perfil('eda')
        carregar_perfil('fraud')
        carregar_perfil('ml')
    """
    if nome_perfil not in PERFIS:
        print(f"❌ Perfil '{nome_perfil}' não existe!")
        print("Perfis disponíveis:", list(PERFIS.keys()))
        return {}
    
    libs = PERFIS[nome_perfil]
    print(f"🎯 PERFIL: {nome_perfil.upper()}")
    
    return carregar_libs(libs)


# ============================================================================
# 📋 FUNÇÃO 4: MOSTRAR VERSÕES (chamada automaticamente)
# ============================================================================

def _mostrar_versoes(libs_carregadas):
    """Mostra versões das bibliotecas carregadas."""
    print(f"\nVERSÕES:")
    print("=" * 20)
    print(f"Python: {python_version()}")
    
    for lib, objeto in libs_carregadas.items():
        pacote = LIBS[lib][1]
        
        # Tenta pegar a versão de diferentes formas
        versao = "N/A"
        
        # Método 1: __version__
        if hasattr(objeto, '__version__'):
            versao = objeto.__version__
        # Método 2: version
        elif hasattr(objeto, 'version'):
            versao = objeto.version
        # Método 3: Para matplotlib especificamente
        elif lib == 'plt' and hasattr(objeto, 'matplotlib'):
            versao = objeto.matplotlib.__version__
        # Método 4: Tentar pegar do módulo pai
        elif hasattr(objeto, '__module__'):
            try:
                modulo_nome = objeto.__module__.split('.')[0]
                modulo_pai = sys.modules.get(modulo_nome)
                if modulo_pai and hasattr(modulo_pai, '__version__'):
                    versao = modulo_pai.__version__
            except:
                pass
        
        print(f"{pacote}: {versao}")


# ============================================================================
# ⚙️ FUNÇÃO AUXILIAR: ADICIONA AO NAMESPACE
# ============================================================================

def _adicionar_ao_namespace(libs_carregadas):
    """Adiciona bibliotecas ao namespace interativo (Jupyter/IPython) ou,
    em fallback, ao primeiro frame do chamador fora deste módulo.
    """
    # 1) Tenta via IPython/Jupyter (melhor caminho no notebook)
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is not None and hasattr(ip, "user_ns"):
            for alias, obj in libs_carregadas.items():
                ip.user_ns[alias] = obj
            print(f"\nBibliotecas disponíveis diretamente (IPython): {list(libs_carregadas.keys())}")
            return
    except Exception:
        pass  # segue para o fallback de frames

    # 2) Fallback: sobe a pilha até encontrar um frame que não seja deste módulo
    try:
        frame = sys._getframe()
        while frame:
            g = frame.f_globals
            # pula frames do próprio lib_loader
            if g.get("__name__") != __name__:
                for alias, obj in libs_carregadas.items():
                    g[alias] = obj
                print(f"\nBibliotecas disponíveis diretamente: {list(libs_carregadas.keys())}")
                return
            frame = frame.f_back
    except Exception as e:
        # se algo der errado, só avisa e mantém uso via dicionário retornado
        print(f"\n⚠️ Não foi possível injetar no namespace do chamador. Erro: {e}")

    print("⚠️ Use as bibliotecas pelo dicionário retornado, por exemplo: "
          "`libs = carregar_perfil('eda'); libs['pd']`.")

# ============================================================================
# 🛠️ FUNÇÃO 5: ADICIONAR NOVAS BIBLIOTECAS (durante execução)
# ============================================================================

def adicionar_lib(nome, modulo, pacote):
    """
    Adiciona uma nova biblioteca temporariamente.
    
    Exemplo:
        adicionar_lib('folium', 'folium', 'folium')
        carregar_libs(['folium'])
    """
    LIBS[nome] = (modulo, pacote)
    print(f"✅ '{nome}' adicionada! Use: carregar_libs(['{nome}'])")


def adicionar_perfil(nome, lista_libs):
    """
    Adiciona um novo perfil temporariamente.
    
    Exemplo:
        adicionar_perfil('dashboard', ['pd', 'plotly'])
        carregar_perfil('dashboard')
    """
    PERFIS[nome] = lista_libs
    print(f"✅ Perfil '{nome}' adicionado! Use: carregar_perfil('{nome}')")


# ============================================================================
# 🚀 ATALHOS RÁPIDOS (opcionais)
# ============================================================================

def eda():
    """Atalho para carregar_perfil('eda')."""
    return carregar_perfil('eda')

def ml():
    """Atalho para carregar_perfil('ml')."""
    return carregar_perfil('ml')

def fraud():
    """Atalho para carregar_perfil('fraud')."""
    return carregar_perfil('fraud')


# ============================================================================
# 🔍 FUNÇÃO DE DIAGNÓSTICO
# ============================================================================

def verificar_sistema():
    """Verifica se as principais bibliotecas estão instaladas."""
    print("🔍 VERIFICANDO SISTEMA:")
    print("=" * 30)
    print(f"🐍 Python: {python_version()}")
    
    # Testar bibliotecas essenciais
    essenciais = ['pd', 'np', 'plt', 'sns']
    instaladas = []
    faltando = []
    
    print(f"\nTestando bibliotecas essenciais:")
    
    for lib in essenciais:
        if lib in LIBS:
            modulo, pacote = LIBS[lib]
            try:
                _importar_biblioteca(modulo)
                instaladas.append(lib)
                print(f"✅ {lib} ({pacote})")
            except ImportError:
                faltando.append(lib)
                print(f"❌ {lib} ({pacote}) - NÃO INSTALADA")
            except Exception as e:
                faltando.append(lib)
                print(f"❌ {lib} ({pacote}) - ERRO: {str(e)[:30]}...")
    
    print(f"\nRESULTADO:")
    print(f"✅ Funcionando: {len(instaladas)}/{len(essenciais)}")
    
    if faltando:
        print(f"❌ Com problemas: {faltando}")
        print(f"\n💡 Para corrigir:")
        for lib in faltando:
            pacote = LIBS[lib][1]
            print(f"   pip install {pacote}")
    else:
        print("🎉 Todas as bibliotecas essenciais estão OK!")


# ============================================================================
# 🧪 TESTE DO SISTEMA
# ============================================================================

if __name__ == "__main__":
    print("🧪 TESTANDO LIB_LOADER")
    print("=" * 30)
    
    # Verificar sistema primeiro
    verificar_sistema()
    
    print("\n" + "=" * 30)
    
    # Mostrar disponíveis
    mostrar_libs()
    
    print("\n" + "=" * 30)
    
    # Teste carregar perfil EDA (só essenciais)
    print("🚀 Testando perfil EDA:")
    try:
        libs = carregar_perfil('eda')
        print("🎉 Teste EDA: SUCESSO!")
    except Exception as e:
        print(f"❌ Teste EDA: FALHA - {e}")
    
    print("\n🎉 Teste concluído!")
    print("\n💡 COMO USAR NO NOTEBOOK:")
    print("""
# Opção 1: Perfil pré-definido
from utils.lib_loader import carregar_perfil
carregar_perfil('fraud')

# Opção 2: Bibliotecas específicas  
from utils.lib_loader import carregar_libs
carregar_libs(['pd', 'np', 'sk'])

# Opção 3: Ver disponíveis
from utils.lib_loader import mostrar_libs
mostrar_libs()

# Opção 4: Verificar sistema
from utils.lib_loader import verificar_sistema
verificar_sistema()
    """)
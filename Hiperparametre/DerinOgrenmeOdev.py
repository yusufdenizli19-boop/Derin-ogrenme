import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ==========================================
# SABİTLER VE SEED AYARI (Okul No Son 2: 26)
# ==========================================
SEED = 26
EPOCHS = 300
BATCH_SIZE = 32

# Veri Kümesi Üretimi (-2 <= p <= 2)
p = np.linspace(-2, 2, 400)
y = 1 + np.sin(np.pi * p / 4)
X_train = p.reshape(-1, 1)
y_train = y.reshape(-1, 1)

print(f"Veri seti üretildi (SEED={SEED}). Tüm ödev deneyleri başlatılıyor...\n")


# ==========================================
# YARDIMCI FONKSİYONLAR
# ==========================================
INITIALIZERS = {
    'RandomNormal':  lambda: keras.initializers.RandomNormal(seed=SEED),
    'RandomUniform': lambda: keras.initializers.RandomUniform(seed=SEED),
    'GlorotUniform': lambda: keras.initializers.GlorotUniform(seed=SEED),
    'HeNormal':      lambda: keras.initializers.HeNormal(seed=SEED),
}

OPTIMIZERS = {
    'SGD':     lambda lr: keras.optimizers.SGD(learning_rate=lr),
    'RMSprop': lambda lr: keras.optimizers.RMSprop(learning_rate=lr),
    'ADAM':    lambda lr: keras.optimizers.Adam(learning_rate=lr),
}


class MetricsPerEpoch(keras.callbacks.Callback):
    """Her epoch sonunda ağ çıkışından RMSE ve R2'yi hesaplar.
    Böylece RMSE ve R2 aynı modelden, aynı tahminlerden elde edilir."""

    def __init__(self, X, y):
        super().__init__()
        self.X, self.y = X, y
        self.rmse, self.r2 = [], []

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model(self.X, training=False).numpy()
        self.rmse.append(float(np.sqrt(np.mean((self.y - pred) ** 2))))
        self.r2.append(float(r2_score(self.y, pred)))


def build_model(cfg):
    """cfg: activation, init, n_layers, n_neurons"""
    init_factory = INITIALIZERS[cfg['init']]
    model_layers = [layers.Input(shape=(1,))]
    for _ in range(cfg['n_layers']):
        model_layers.append(layers.Dense(cfg['n_neurons'],
                                         activation=cfg['activation'],
                                         kernel_initializer=init_factory()))
    model_layers.append(layers.Dense(1, activation='linear',
                                     kernel_initializer=init_factory()))
    return keras.Sequential(model_layers)


def train_and_eval(cfg):
    """Verilen konfigürasyonla modeli eğitir, sonuçları döndürür."""
    # Her model aynı seed ile başlar -> adil karşılaştırma
    keras.utils.set_random_seed(SEED)

    model = build_model(cfg)
    model.compile(optimizer=OPTIMIZERS[cfg['optimizer']](cfg['lr']),
                  loss='mean_squared_error')
    cb = MetricsPerEpoch(X_train, y_train)

    start = time.time()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              verbose=0, callbacks=[cb])
    elapsed = time.time() - start  # (callback süresi de dahildir)

    y_pred = model.predict(X_train, verbose=0)
    return {
        'rmse': float(np.sqrt(np.mean((y_train - y_pred) ** 2))),
        'r2': float(r2_score(y_train, y_pred)),
        'time': elapsed,
        'rmse_hist': cb.rmse,
        'r2_hist': cb.r2,
        'y_pred': y_pred,
    }


def run_experiment(step_title, col_title, short_title, base_cfg, key, variants,
                   label_fn=str, fname=None):
    """Tek bir parametreyi değiştirerek deney yapar.
    variants: {gösterilecek_isim: parametre_değeri}
    En iyi model (en düşük RMSE) otomatik seçilir ve grafikleri çizilir.
    Döndürür: (en iyi isim, en iyi parametre değeri)"""
    print(f"=== {step_title} ===")
    print(f"{col_title:<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
    print("-" * 55)

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, value in variants.items():
        cfg = dict(base_cfg)
        cfg[key] = value
        res = train_and_eval(cfg)
        results[name] = res
        print(f"{name:<15} | {res['rmse']:<10.5f} | {res['r2']:<10.5f} | {res['time']:<10.2f}")
        axes[0].plot(res['rmse_hist'], linewidth=2, label=label_fn(name))

    best_name = min(results, key=lambda k: results[k]['rmse'])
    best = results[best_name]
    print(f"--> En iyi: {best_name} (RMSE={best['rmse']:.5f}, R2={best['r2']:.5f})\n")

    axes[0].set_yscale('log')
    axes[0].set_title(f"{short_title} / RMSE Error (Log)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (RMSE)")
    axes[0].legend()
    axes[0].grid(True, which="both", ls="--")

    axes[1].plot(p, y, label='Hedef (Target)', color='midnightblue', linewidth=2)
    axes[1].plot(p, best['y_pred'], label=f'Ağ Çıkışı ({best_name})',
                 color='orangered', linestyle='--', linewidth=2)
    axes[1].set_title(f"Hedef vs Ağ Çıkışı ({best_name})")
    axes[1].set_xlabel("Giriş (p)")
    axes[1].set_ylabel("Çıkış (y)")
    axes[1].legend()
    axes[1].grid(True, ls="--")

    axes[2].plot(range(1, EPOCHS + 1), best['r2_hist'], color='forestgreen', linewidth=2)
    axes[2].set_title(f"Epoch / R2 Skoru ({best_name})")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R2 Score")
    axes[2].grid(True, ls="--")

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()

    return best_name, variants[best_name]


# Başlangıç konfigürasyonu (her adımda en iyi değer bir sonraki adıma aktarılır)
cfg = {
    'activation': 'sigmoid',
    'lr': 0.1,
    'optimizer': 'SGD',
    'init': 'RandomNormal',
    'n_layers': 1,
    'n_neurons': 2,
}

# ==========================================
# 1. ADIM: AKTİVASYON FONKSİYONLARI DENEYİ
# ==========================================
activations = ['linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'silu']
_, cfg['activation'] = run_experiment(
    "1. AKTİVASYON FONKSİYONLARI DEĞİŞİMİ", "Aktivasyon", "Aktivasyon",
    cfg, 'activation', {a: a for a in activations}, fname="adim1_aktivasyon.png")

# ==========================================
# 2. ADIM: ÖĞRENME ORANI (LR) DENEYİ
# ==========================================
lrs = {str(lr): lr for lr in [0.1, 0.01, 0.001]}
_, cfg['lr'] = run_experiment(
    "2. ÖĞRENME ORANI DEĞİŞİMİ", "Öğrenme Oranı", "Öğrenme Oranı",
    cfg, 'lr', lrs, label_fn=lambda n: f'LR={n}', fname="adim2_ogrenme_orani.png")

# ==========================================
# 3. ADIM: OPTİMİZASYON YÖNTEMLERİ DENEYİ
# ==========================================
_, cfg['optimizer'] = run_experiment(
    "3. OPTİMİZASYON YÖNTEMLERİ DEĞİŞİMİ", "Optimizasyon", "Optimizasyon",
    cfg, 'optimizer', {n: n for n in OPTIMIZERS}, fname="adim3_optimizasyon.png")

# ==========================================
# 4. ADIM: AĞIRLIK BAŞLANGIÇ DAĞILIMLARI
# ==========================================
_, cfg['init'] = run_experiment(
    "4. AĞIRLIK BAŞLANGIÇ DAĞILIMLARI DEĞİŞİMİ", "Dağılım", "Initializer",
    cfg, 'init', {n: n for n in INITIALIZERS}, fname="adim4_initializer.png")

# ==========================================
# 5. ADIM: GİZLİ KATMAN SAYISI DENEYİ
# ==========================================
layer_counts = {f'{c} Katman': c for c in [1, 2, 5, 10, 20]}
_, cfg['n_layers'] = run_experiment(
    "5. GİZLİ KATMAN SAYISI DEĞİŞİMİ", "Katman Sayısı", "Gizli Katman",
    cfg, 'n_layers', layer_counts, fname="adim5_katman_sayisi.png")

# ==========================================
# 6. ADIM: GİZLİ KATMANDAKİ NÖRON SAYISI DENEYİ
# ==========================================
neuron_counts = {f'{n} Nöron': n for n in [2, 5, 10, 20, 50]}
_, cfg['n_neurons'] = run_experiment(
    "6. GİZLİ KATMANDAKİ NÖRON SAYISI DEĞİŞİMİ", "Nöron Sayısı", "Nöron Sayısı",
    cfg, 'n_neurons', neuron_counts, fname="adim6_noron_sayisi.png")

print("Seçilen nihai konfigürasyon:", cfg)

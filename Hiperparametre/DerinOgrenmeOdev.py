import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

try:                                   # Colab'da tabloyu güzel göstermek için
    from IPython.display import display
except ImportError:
    display = print

# ==========================================
# SABİTLER VE SEED AYARI (Okul No Son 2: 26)
# ==========================================
SEED = 26
EPOCHS = 300
BATCH_SIZE = 64

# FREQ=1: neredeyse doğrusal (çok kolay) | FREQ=2: 1 tam periyot (2 nöronla zor, 10+ nöronla öğrenilir)
# FREQ=4: 2 periyot | FREQ=8: 4 periyot (bu seviyede küçük ağlar hiç öğrenemez)
FREQ = 2

p = np.linspace(-2, 2, 400)
y = 1 + np.sin(FREQ * np.pi * p / 4)
X_train = p.reshape(-1, 1)
y_train = y.reshape(-1, 1)

print(f"Veri seti üretildi (SEED={SEED}, FREQ={FREQ}). Tüm ödev deneyleri başlatılıyor...\n")

# ==========================================
# YARDIMCI FONKSİYONLAR
# ==========================================
INITIALIZERS = {
    'RandomNormal':  lambda: keras.initializers.RandomNormal(seed=SEED),
    'RandomUniform': lambda: keras.initializers.RandomUniform(seed=SEED),
    'GlorotUniform': lambda: keras.initializers.GlorotUniform(seed=SEED),
    'HeNormal':      lambda: keras.initializers.HeNormal(seed=SEED),
}

# Öğrenme oranı epoch boyunca cosine ile lr -> lr*0.01 düşer: sabit lr yüzünden
# oluşan titremeyi (özellikle geniş/derin ağlarda) söndürür.
# clipnorm=1.0: gradyan patlamasını engeller.
STEPS_PER_EPOCH = int(np.ceil(len(X_train) / BATCH_SIZE))
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH


def lr_schedule(lr):
    # lr -> lr*0.001: sona doğru adımlar çok küçülür, titreme söner
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=TOTAL_STEPS, alpha=0.001)


# RMSprop/Adam gradyanı normalize ettiği için aynı lr'de SGD'den çok daha büyük
# etkili adım atar (adım ~ lr, gradyan küçülse bile). Bu yüzden SGD ile seçilen lr'yi
# bu optimizer'larda küçültüyoruz. İsterseniz değerleri değiştirin.
LR_SCALE = {'SGD': 1.0, 'RMSprop': 0.05, 'ADAM': 0.2}

OPTIMIZERS = {
    'SGD':     lambda lr: keras.optimizers.SGD(
        learning_rate=lr_schedule(lr * LR_SCALE['SGD']), clipnorm=1.0),
    'RMSprop': lambda lr: keras.optimizers.RMSprop(
        learning_rate=lr_schedule(lr * LR_SCALE['RMSprop']), rho=0.99, clipnorm=1.0),
    'ADAM':    lambda lr: keras.optimizers.Adam(
        learning_rate=lr_schedule(lr * LR_SCALE['ADAM']), clipnorm=1.0),
}

REF_WIDTH = 10   # bu nöron sayısına kadar lr ölçeklenmez

ALL_ROWS = []   # tüm deneylerin sonuçları (tablo için)


class MetricsPerEpoch(keras.callbacks.Callback):
    """Her epoch sonunda ağ çıkışından RMSE ve R2'yi hesaplar."""

    def __init__(self, X, y):
        super().__init__()
        self.X, self.y = X, y
        self.rmse, self.r2 = [], []

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model(self.X, training=False).numpy()
        self.rmse.append(float(np.sqrt(np.mean((self.y - pred) ** 2))))
        self.r2.append(float(r2_score(self.y, pred)))


def build_model(cfg):
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
    keras.utils.set_random_seed(SEED)

    model = build_model(cfg)
    # Geniş katmanlarda çıkış, çok sayıda ağırlığın toplamı olduğu için aynı lr'de
    # daha büyük sıçrar. Bu yüzden n_neurons > REF_WIDTH ise lr ters orantılı küçültülür.
    eff_lr = cfg['lr'] * min(1.0, REF_WIDTH / cfg['n_neurons'])
    model.compile(optimizer=OPTIMIZERS[cfg['optimizer']](eff_lr),
                  loss='mean_squared_error')
    cb = MetricsPerEpoch(X_train, y_train)

    # En iyi epoch'taki ağırlıklara geri dön (patience=EPOCHS -> erken durdurma yok).
    # Böylece eğitim sonunda kötüleşen/sapan modeller yüzünden negatif R2 çıkmaz.
    best_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=EPOCHS,
                                            restore_best_weights=True)
    nan_cb = keras.callbacks.TerminateOnNaN()

    start = time.time()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              verbose=0, callbacks=[cb, best_cb, nan_cb])
    elapsed = time.time() - start

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
        ALL_ROWS.append({
            'Adım': short_title, 'Değer': str(name),
            'RMSE': res['rmse'], 'R2': res['r2'], 'Süre (sn)': res['time'],
        })

    best_name = min(results, key=lambda k: results[k]['rmse'])
    best = results[best_name]
    print(f"--> En iyi: {best_name} (RMSE={best['rmse']:.5f}, R2={best['r2']:.5f})\n")
    for row in ALL_ROWS:
        if row['Adım'] == short_title:
            row['En İyi'] = '★' if row['Değer'] == str(best_name) else ''

    n_ep = len(best['r2_hist'])   # NaN nedeniyle erken biten eğitimlere karşı

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

    one_minus_r2 = np.maximum(1 - np.array(best['r2_hist']), 1e-12)
    axes[2].plot(range(1, n_ep + 1), one_minus_r2, color='forestgreen', linewidth=2)
    axes[2].set_yscale('log')
    axes[2].set_title(f"Epoch / 1 - R2 ({best_name})")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("1 - R2 (log)")
    axes[2].grid(True, which="both", ls="--")

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()

    return best_name, variants[best_name]


# Başlangıç konfigürasyonu (her adımın en iyisi sonraki adıma aktarılır)
cfg = {
    'activation': 'sigmoid',
    'lr': 0.01,
    'optimizer': 'SGD',
    'init': 'RandomNormal',
    'n_layers': 1,
    'n_neurons': 2,
}

# ==========================================
# 1. ADIM: AKTİVASYON FONKSİYONLARI
# ==========================================
activations = ['linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'silu']
_, cfg['activation'] = run_experiment(
    "1. AKTİVASYON FONKSİYONLARI DEĞİŞİMİ", "Aktivasyon", "Aktivasyon",
    cfg, 'activation', {a: a for a in activations}, fname="adim1_aktivasyon.png")

# ==========================================
# 2. ADIM: ÖĞRENME ORANI
# ==========================================
lrs = {str(lr): lr for lr in [0.05, 0.01, 0.005]}
_, cfg['lr'] = run_experiment(
    "2. ÖĞRENME ORANI DEĞİŞİMİ", "Öğrenme Oranı", "Öğrenme Oranı",
    cfg, 'lr', lrs, label_fn=lambda n: f'LR={n}', fname="adim2_ogrenme_orani.png")

# ==========================================
# 3. ADIM: OPTİMİZASYON YÖNTEMLERİ
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
# 5. ADIM: GİZLİ KATMAN SAYISI
# ==========================================
layer_counts = {f'{c} Katman': c for c in [1, 2, 5, 10, 20]}
_, cfg['n_layers'] = run_experiment(
    "5. GİZLİ KATMAN SAYISI DEĞİŞİMİ", "Katman Sayısı", "Gizli Katman",
    cfg, 'n_layers', layer_counts, fname="adim5_katman_sayisi.png")

# ==========================================
# 6. ADIM: NÖRON SAYISI
# ==========================================
neuron_counts = {f'{n} Nöron': n for n in [2, 5, 10, 20, 50]}
_, cfg['n_neurons'] = run_experiment(
    "6. GİZLİ KATMANDAKİ NÖRON SAYISI DEĞİŞİMİ", "Nöron Sayısı", "Nöron Sayısı",
    cfg, 'n_neurons', neuron_counts, fname="adim6_noron_sayisi.png")

print("Seçilen nihai konfigürasyon:", cfg)

# ==========================================
# SONUÇ TABLOLARI
# ==========================================
df = pd.DataFrame(ALL_ROWS)[['Adım', 'Değer', 'RMSE', 'R2', 'Süre (sn)', 'En İyi']]
df['RMSE'] = df['RMSE'].round(5)
df['R2'] = df['R2'].round(5)
df['Süre (sn)'] = df['Süre (sn)'].round(2)

print("\n=== TÜM DENEYLERİN SONUÇ TABLOSU ===")
display(df.style.hide(axis='index')
          .apply(lambda r: ['background-color: #d4edda' if r['En İyi'] == '★' else '' for _ in r], axis=1))

summary = df[df['En İyi'] == '★'].drop(columns='En İyi').reset_index(drop=True)
print("\n=== HER ADIMIN EN İYİ SONUCU ===")
display(summary)

df.to_csv("tum_sonuclar.csv", index=False, encoding="utf-8-sig")
summary.to_csv("ozet_sonuclar.csv", index=False, encoding="utf-8-sig")

# Tabloyu görsel olarak da kaydet (rapora koymak için)
fig, ax = plt.subplots(figsize=(10, 0.4 * len(summary) + 1.2))
ax.axis('off')
tbl = ax.table(cellText=summary.values, colLabels=summary.columns, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.5)
plt.savefig("ozet_tablo.png", dpi=200, bbox_inches='tight')
plt.show()

# Colab'dan indirmek için (isteğe bağlı):
# from google.colab import files
# for f in ["tum_sonuclar.csv", "ozet_sonuclar.csv", "ozet_tablo.png"]:
#     files.download(f)

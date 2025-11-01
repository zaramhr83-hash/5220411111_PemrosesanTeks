# ============================================================
# 1️⃣ IMPORT LIBRARY
# ============================================================
# pip install pandas tqdm Sastrawi nltk wordcloud matplotlib emoji
import os
import pandas as pd
from tqdm import tqdm
import re
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================================================
# 2️⃣ LOAD DATASET (TANPA SCRAPING)
# ============================================================
# Replace this path with your actual dataset location
file_path = '/mnt/data/dataTiktok.xlsx'
xls = pd.ExcelFile(file_path)
df_tiktok = pd.read_excel(xls, 'Sheet1')

print(f"✅ Dataset berhasil dimuat: {len(df_tiktok)} data")

# Hapus duplikat & kosong
df_tiktok.drop_duplicates(subset='text', inplace=True)
df_tiktok.dropna(subset=['text'], inplace=True)
df_tiktok.reset_index(drop=True, inplace=True)

# ============================================================
# 3️⃣ PREPROCESSING (CLEANING, NORMALISASI, STOPWORDS, BIGRAM/TRIGRAM)
# ============================================================
print("\n🧹 Tahap 1: Preprocessing dimulai...")

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Stopwords dasar dari NLTK
stop_words = set(stopwords.words('indonesian'))

# ============================================================
# Tambahan Stopwords Kustom untuk Bahasa Gaul / Umum Tak Bermakna
# ============================================================
custom_stopwords = {
    'nya','sip','oke','ok','yah','ya','lah','deh','dong','sih','nih','loh',
    'pls','please','ges','bro','guys','bang','anj','anjir','anjay','wkwk',
    'gua','gw','aku','saya','kamu','anda','loe','lu','tpi','tp','pas',
    'gitu','kalo','kalau','kayak','yg','yaudah','udah','dah','aja',
    'buka','nama','pakai','pake','lagi','doang','nih','tuh','bgt','banget',
    'gpp','mantap','mantul','bikin','kan','itu','ini','dong','lah'
}
stop_words = stop_words.union(custom_stopwords)

# Kamus normalisasi slang & koreksi typo
slang_dict = { 
    'aja':'saja','aku':'saya','apknya':'aplikasi','bagu':'bagus','bener':'benar','bgs':'bagus','bgt':'banget',
    'bikin':'buat','blm':'belum','bngt':'banget','dgn':'dengan','dpt':'dapat','ga':'tidak','gaje':'tidak jelas',
    'gak':'tidak','gamenya':'game','gk':'tidak','jele':'jelek','kamu':'anda','km':'kamu','krn':'karena',
    'laggy':'lambat','mainn':'main','makasih':'terima kasih','mksih':'terima kasih','ngelag':'lambat',
    'ngecrash':'crash','ngehang':'macet','nggak':'tidak','nih':'ini','parah':'buruk sekali','ser':'seru',
    'suk':'suka','tdk':'tidak','tp':'tapi','trs':'terus','udh':'sudah','yg':'yang','errornya':'error'
}

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text)

    # 1️⃣ Hilangkan URL, mention, angka, emoji
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = emoji.replace_emoji(text, replace='')

    # 2️⃣ Negasi handling: ubah "tidak bagus" jadi "tidak_bagus"
    text = re.sub(r"tidak (\w+)", r"tidak_\1", text)

    # 3️⃣ Bersihkan simbol dan ubah huruf kecil
    text = re.sub(r"[^a-zA-Z_\s]", " ", text).lower()

    # 4️⃣ Tokenizing
    tokens = word_tokenize(text)

    # 5️⃣ Normalisasi slang
    tokens = [slang_dict.get(w, w) for w in tokens]

    # 6️⃣ Stopword removal
    filtered = [w for w in tokens if w not in stop_words and len(w) > 3]

    # 7️⃣ Stemming
    stemmed = [stemmer.stem(w) for w in filtered]

    # 8️⃣ Bigram & Trigram
    bigrams = ['_'.join(bg) for bg in ngrams(stemmed, 2)] if len(stemmed) >= 2 else []
    trigrams = ['_'.join(tg) for tg in ngrams(stemmed, 3)] if len(stemmed) >= 3 else []
    all_terms = stemmed + bigrams + trigrams

    return " ".join(all_terms)

tqdm.pandas(desc="Cleaning Text")
df_tiktok["cleaned"] = df_tiktok["text"].progress_apply(clean_text)
df_tiktok = df_tiktok[df_tiktok["cleaned"].str.strip() != ""]

df_tiktok.to_csv("dataTiktok_cleaned.csv", index=False, encoding="utf-8")
print("✅ Tahap 1 selesai — data bersih disimpan ke 'dataTiktok_cleaned.csv'")

# ============================================================
# 4️⃣ LABELING SENTIMEN BERDASARKAN TEKS (Optional)
# ============================================================
# For simplicity, label sentiment based on the number of likes or engagement (optional)
def label_sentimen(diggCount):
    if diggCount >= 10000:
        return "Positif"
    elif diggCount == 5000:
        return "Netral"
    else:
        return "Negatif"

df_tiktok["sentimen"] = df_tiktok["diggCount"].apply(label_sentimen)
df_tiktok.to_csv("dataTiktok_labeled.csv", index=False, encoding="utf-8")
print("✅ Labeling selesai — file 'dataTiktok_labeled.csv' tersimpan")

# ============================================================
# 5️⃣ EDA (Exploratory Data Analysis)
# ============================================================
print("\n📊 Tahap 3: EDA...")

# Statistik umum rating
print(df_tiktok["diggCount"].describe())

# Distribusi like count
plt.figure(figsize=(7,5))
df_tiktok["diggCount"].value_counts().sort_index().plot(kind="bar", color="skyblue", title="Distribusi Like Count")
plt.xlabel("Jumlah Likes")
plt.ylabel("Jumlah Video")
plt.show()

# Panjang ulasan
df_tiktok["length"] = df_tiktok["text"].apply(lambda x: len(str(x).split()))
print("Rata-rata panjang ulasan:", round(df_tiktok["length"].mean(),2))

# Scatter: panjang ulasan vs likes
plt.figure(figsize=(6,4))
plt.scatter(df_tiktok["diggCount"], df_tiktok["length"], alpha=0.3, color="orange")
plt.title("Hubungan Panjang Ulasan vs Jumlah Likes")
plt.xlabel("Jumlah Likes")
plt.ylabel("Panjang (kata)")
plt.show()

# ============================================================
# 6️⃣ WORDCLOUD PER SENTIMEN
# ============================================================
print("\n🎨 Tahap 4: WordCloud...")

def generate_wordcloud(text, title):
    plt.figure(figsize=(8,6))
    wc = WordCloud(width=800, height=600, background_color="white", colormap="viridis", max_words=150).generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.show()

text_all = " ".join(df_tiktok["cleaned"])
text_pos = " ".join(df_tiktok[df_tiktok["sentimen"]=="Positif"]["cleaned"])
text_net = " ".join(df_tiktok[df_tiktok["sentimen"]=="Netral"]["cleaned"])
text_neg = " ".join(df_tiktok[df_tiktok["sentimen"]=="Negatif"]["cleaned"])

generate_wordcloud(text_all, "WordCloud Semua Ulasan")
generate_wordcloud(text_pos, "WordCloud Ulasan Positif")
generate_wordcloud(text_net, "WordCloud Ulasan Netral")
generate_wordcloud(text_neg, "WordCloud Ulasan Negatif")

print("\n✅ Semua tahap analisis deskriptif selesai!")

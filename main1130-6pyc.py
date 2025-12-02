# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pycaret.classification import *
from sklearn.preprocessing import MinMaxScaler

plt.rcParams["font.family"] = "Meiryo"

# ---------------------------------------------------------
# 銘柄・期間
ticker = "8035.T"
start = "2024-12-02"
end   = "2025-11-28"

# 株価データ取得
df_stock = yf.download(ticker, start=start, end=end)[['Open','High','Low','Close','Volume']]
df_stock["Diff"] = df_stock["Close"].diff().fillna(0)
df_stock["SMA2"] = df_stock["Close"].rolling(2).mean().fillna(method='bfill')

# 米国株（S&P500）
df_us = yf.download("^GSPC", start=start, end=end)[["Close"]]
df_us["US_Diff"] = df_us["Close"].diff().shift(1)
df_us["US_Pct"]  = df_us["Close"].pct_change().shift(1)
df_us.fillna(0, inplace=True)
df_stock = df_stock.merge(df_us[["US_Diff","US_Pct"]],
                          left_index=True, right_index=True, how="left").fillna(0)

# ニュース（区間全体に分散配置）
news_texts = [
    "半導体装置の売上が増加する見込みです",
    "市場の需要が高まっています",
    "新製品発表で業績改善が期待されます",
    "東京エレクトロンの株価が上昇しました",
    "業績好調で株価が上昇傾向です",
    "海外需要の回復が見込まれます",
    "生産能力の増強計画が発表されました"
]
news_dates = pd.to_datetime([
    "2024-12-05","2025-01-10","2025-03-15",
    "2025-05-20","2025-07-25","2025-09-30","2025-11-20"
])
news_labels = ["A","B","C","D","E","F","G"]

df_news = pd.DataFrame({"text": news_texts}, index=news_dates)

# TF-IDF（助詞フィルター導入）
tokenizer = Tokenizer()
stopwords = {"傾向","上昇","れ","まし","ます","は","が","を","に","で","と","も","の","へ","から","より"}

def tokenize(text):
    return " ".join([
        t.surface for t in tokenizer.tokenize(text)
        if t.surface not in stopwords
    ])

df_news["token"] = df_news["text"].apply(tokenize)

vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, token_pattern=r"(?u)\b\w+\b")
tfidf_features = vectorizer.fit_transform(df_news["token"]).toarray()

tfidf_df = pd.DataFrame(tfidf_features, index=df_news.index,
                        columns=[f"tfidf_{i}" for i in range(tfidf_features.shape[1])])

for col in tfidf_df.columns:
    df_stock[col] = df_stock.index.map(lambda d: tfidf_df.loc[d, col] if d in tfidf_df.index else 0)

# 予測ターゲット
df_stock["target"] = (df_stock["Close"].shift(-1) > df_stock["Close"]).astype(int)
df_model = df_stock.dropna()

# MultiIndex解除
df_model.columns = [
    '_'.join([str(c) for c in col if c]) if isinstance(col, tuple) else str(col)
    for col in df_model.columns
]

# PyCaret セットアップ（fix_imbalance追加）
clf_setup = setup(
    data=df_model,
    target="target",
    session_id=42,
    silent=True,
    verbose=False,
    normalize=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.95,
    fix_imbalance=True  # ★追加
)

# RFモデル（class_weight追加）
rf_model = create_model('rf', class_weight='balanced')  # ★追加
final_model = finalize_model(rf_model)

# キャリブレーション（確率補正）
cal_model = calibrate_model(final_model, method='sigmoid')  # ★追加

# 予測結果
pred = predict_model(cal_model, data=df_model)
df_pred = pred[["Label","Score"]].copy()
df_pred.index = df_model.index
df_pred["Prob_Up"]   = df_pred["Score"]
df_pred["Prob_Down"] = 1 - df_pred["Score"]

# 特徴量重要度
features = df_model.drop(columns=["target"]).columns
importances = final_model.feature_importances_
min_len = min(len(features), len(importances))
features = features[:min_len]
importances = importances[:min_len]

feat_imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

jp_map = {
    "Open_8035.T":"始値", "High_8035.T":"高値", "Low_8035.T":"安値",
    "Close_8035.T":"終値", "Volume_8035.T":"出来高", "SMA2":"短期平均",
    "US_Diff":"米国差分", "US_Pct":"米国変化率"
}
feat_imp_df["Feature"] = feat_imp_df["Feature"].apply(lambda x: jp_map[x] if x in jp_map else x)

scaler = MinMaxScaler()
feat_imp_df["Scaled"] = scaler.fit_transform(feat_imp_df[["Importance"]])
feat_imp_df["Scaled"] = feat_imp_df["Scaled"].apply(lambda x: max(x,0.1))

# 特徴量重要度グラフ（TF-IDF全項目右に文字列表示）
plt.figure(figsize=(10,6))
plt.barh(feat_imp_df["Feature"], feat_imp_df["Scaled"])
plt.title("特徴量重要度（日本語ラベル＋TF-IDFテキスト付き） (2024-12-02〜2025-11-28)")
plt.xlabel("重要度（0〜1）")

try:
    feature_names = vectorizer.get_feature_names_out()
except AttributeError:
    feature_names = vectorizer.get_feature_names()

for i, (feature, value) in enumerate(zip(feat_imp_df["Feature"], feat_imp_df["Scaled"])):
    if feature.startswith("tfidf_"):
        idx = int(feature.split("_")[1])
        if idx < len(news_texts):
            label_text = news_texts[idx]
        elif idx < len(feature_names):
            label_text = feature_names[idx]
        else:
            label_text = f"TF-IDF{idx}"
        plt.text(value + 0.02, i, label_text, fontsize=8, va='center')

plt.tight_layout()
plt.show()

# 株価推移
plt.figure(figsize=(10,4))
plt.plot(df_model.index, df_model["Close_8035.T"], marker="o", label="終値")
plt.plot(df_model.index, df_model["SMA2"],  marker="x", label="短期平均")
plt.title("株価推移 (2024-12-02〜2025-11-28)")
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gcf().autofmt_xdate()
plt.xlim(pd.to_datetime("2024-12-02"), pd.to_datetime("2025-11-28"))
plt.tight_layout()
plt.show()

# ニュースTF-IDF
plt.figure(figsize=(10,4))
for i, col in enumerate(tfidf_df.columns[:7]):
    plt.bar(news_labels[i], df_model[col].max(), label=f"{news_labels[i]}: {news_texts[i]}")
plt.title("ニュース TF-IDF（最大値） (2024-12-02〜2025-11-28)")
plt.ylabel("TF-IDF")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# 翌日の上昇／下落確率（期間を2024/12/02〜2025-11-28に変更）
plt.figure(figsize=(12,4))
plt.bar(df_pred.index, df_pred["Prob_Up"], label="上昇確率", color="green", alpha=0.6)
plt.bar(df_pred.index, df_pred["Prob_Down"], label="下落確率", color="red", alpha=0.3, bottom=df_pred["Prob_Up"])
plt.title("翌日の株価上昇／下落確率（PyCaret v2.3.10版） (2024-12-02〜2025-11-28)")
plt.ylabel("確率")
plt.legend(loc="upper right", fontsize=10, frameon=True, facecolor="white", edgecolor="black")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.gcf().autofmt_xdate()
plt.xlim(pd.to_datetime("2024-12-02"), pd.to_datetime("2025-11-28"))
plt.tight_layout()
plt.show()
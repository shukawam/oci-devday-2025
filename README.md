# Oracle Developer Days 2025

Oracle Developer Days 2025 - 実践！Datadog で高める OCI のオブザーバビリティで使用したデモアプリケーションです。

## 構成

```sh
.
├── app        # デモアプリケーション本体
├── kubernetes # アプリケーションデプロイ用のKubernetes Manifest
└── setup      # デモアプリケーションで使用するサンプルデータの収集やベクトルデータベースのセットアップ用
```

## 動かし方（ローカル）

`.env.example` をコピーし、 `.env` を作成します。

```sh
# --- .env ---
# Google Cloud
GOOGLE_API_KEY="<Google Cloud API Key>"

# Oracle Cloud Infrastructure
USERNAME="<Username(Oracle Database 23ai)>"
PASSWORD="<Password(Oracle Database 23ai)"
DSN="<DSN(Oracle Database 23ai)>"
COMPARTMENT_ID="<Compartment ID>"
```

アプリケーションを起動します。

```sh
cd app; streamlit run main.py
```

`http://localhost:8501` にアクセスすると起動されたアプリケーションにアクセスすることができます。
